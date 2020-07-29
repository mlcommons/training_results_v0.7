#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import itertools
import os
import math
import torch
import time
import ctypes
import random
import sys
import unicodedata
import six
import re
import gc

from copy import deepcopy
from functools import reduce
from six.moves import xrange
import numpy as np

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils, tokenizer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import dictionary
from fairseq.data import language_pair_dataset

from mlperf_log_utils import log_start, log_end, log_event, barrier
from mlperf_logging.mllog import constants
from mlperf_logging import mllog


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def setup_seeds(master_seed, epochs, device, rank, world_size):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if rank == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            print(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        print(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, world_size)

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds


def main(args):
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)

    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False

    log_start(key=constants.INIT_START, log_all_ranks=True)

    # preinit and warmup streams/groups for allreduce communicators
    allreduce_communicators=None
    if args.distributed_world_size > 1 and args.enable_parallel_backward_allred_opt:
        allreduce_groups = [torch.distributed.new_group() for _ in range(args.parallel_backward_allred_cuda_nstreams)]
        allreduce_streams = [torch.cuda.Stream() for _ in range(args.parallel_backward_allred_cuda_nstreams)]
        for group, stream in zip(allreduce_groups,allreduce_streams):
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(torch.cuda.FloatTensor(1), group=group)
        allreduce_communicators=(allreduce_groups,allreduce_streams)

    if args.max_tokens is None:
        args.max_tokens = 6000

    print(args)

    log_event(key=constants.GLOBAL_BATCH_SIZE, value=args.max_tokens*args.distributed_world_size)
    log_event(key=constants.OPT_NAME, value=args.optimizer)
    assert(len(args.lr) == 1)
    log_event(key=constants.OPT_BASE_LR, value=args.lr[0] if len(args.lr) == 1 else args.lr)
    log_event(key=constants.OPT_LR_WARMUP_STEPS, value=args.warmup_updates)
    assert(args.max_source_positions == args.max_target_positions)
    log_event(key=constants.MAX_SEQUENCE_LENGTH, value=args.max_target_positions, metadata={'method': 'discard'})
    log_event(key=constants.OPT_ADAM_BETA_1, value=eval(args.adam_betas)[0])
    log_event(key=constants.OPT_ADAM_BETA_2, value=eval(args.adam_betas)[1])
    log_event(key=constants.OPT_ADAM_EPSILON, value=args.adam_eps)
    log_event(key=constants.SEED, value=args.seed)

    # L2 Sector Promotion
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    result = ctypes.CDLL('libcudart.so').cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    result = ctypes.CDLL('libcudart.so').cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))

    worker_seeds, shuffling_seeds = setup_seeds(args.seed, args.max_epoch + 1,
                                                torch.device('cuda'),
                                                args.distributed_rank,
                                                args.distributed_world_size,
                                                )
    worker_seed = worker_seeds[args.distributed_rank]
    print(f'Worker {args.distributed_rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Build trainer
    if args.fp16:
        if args.distributed_weight_update != 0:
            from fairseq.fp16_trainer import DistributedFP16Trainer
            trainer = DistributedFP16Trainer(args, task, model, criterion, allreduce_communicators=allreduce_communicators)
        else:
            from fairseq.fp16_trainer import FP16Trainer
            trainer = FP16Trainer(args, task, model, criterion, allreduce_communicators=allreduce_communicators)
    else:
        if torch.cuda.get_device_capability(0)[0] >= 7:
            print('| NOTICE: your device may support faster training with --fp16')

        trainer = Trainer(args, task, model, criterion, allreduce_communicators=None)

    #if (args.online_eval or args.target_bleu) and not args.remove_bpe:
    #    args.remove_bpe='@@ '

    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(args.max_tokens, args.max_sentences, ))

    # Initialize dataloader
    max_positions = trainer.get_model().max_positions()

    # Send a dummy batch to warm the caching allocator
    dummy_batch = language_pair_dataset.get_dummy_batch_isolated(args.max_tokens, max_positions, 8)
    trainer.dummy_train_step(dummy_batch)

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch if args.max_epoch >= 0 else math.inf
    max_update = args.max_update or math.inf
    tgt_bleu = args.target_bleu or math.inf
    current_bleu = 0.0
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]

    # mlperf compliance synchronization
    if args.distributed_world_size > 1:
        assert(torch.distributed.is_initialized())
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()

    log_end(key=constants.INIT_STOP, sync=False)

    log_start(key=constants.RUN_START, sync=True)
    # second sync after RUN_START tag is printed.
    # this ensures no rank touches data until after RUN_START tag is printed.
    barrier()

    # Load dataset splits
    load_dataset_splits(task, ['train', 'test'])

    log_event(key=constants.TRAIN_SAMPLES,
              value=len(task.dataset(args.train_subset)),
              sync=False)
    log_event(key=constants.EVAL_SAMPLES,
              value=len(task.dataset(args.gen_subset)),
              sync=False)

    ctr = 0

    start = time.time()
    epoch_itr = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.enable_dataloader_pin_memory,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seeds=shuffling_seeds,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        epoch=epoch_itr.epoch if ctr is not 0 else 0,
        bucket_growth_factor=args.bucket_growth_factor,
        seq_len_multiple=args.seq_len_multiple,
        batching_scheme=args.batching_scheme,
        batch_multiple_strategy=args.batch_multiple_strategy,
    )
    print("got epoch iterator", time.time() - start)

    # Main training loop
    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update and current_bleu < tgt_bleu:
        first_epoch = epoch_itr.epoch+1
        log_start(key=constants.BLOCK_START,
                     metadata={'first_epoch_num': first_epoch, 'epoch_count': 1},
                     sync=False)
        log_start(key=constants.EPOCH_START, metadata={'epoch_num': first_epoch}, sync=False)

        gc.disable()

        # Load the latest checkpoint if one is available
        if ctr is 0:
            load_checkpoint(args, trainer, epoch_itr)

        # train for one epoch
        start = time.time()
        #exit(1)
        train(args, trainer, task, epoch_itr, shuffling_seeds)
        print("epoch time ", time.time() - start)

        start = time.time()
        log_end(key=constants.EPOCH_STOP, metadata={'epoch_num': first_epoch}, sync=False)

        # Eval BLEU score
        if args.online_eval or (not tgt_bleu is math.inf):
            current_bleu = score(args, trainer, task, epoch_itr, args.gen_subset)
            log_event(key=constants.EVAL_ACCURACY,
                         value=float(current_bleu) / 100.0,
                         metadata={'epoch_num': first_epoch})

        gc.enable()

        # Only use first validation loss to update the learning rate
        #lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # Save checkpoint
        #if epoch_itr.epoch % args.save_interval == 0:
        #    save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        ctr = ctr + 1
        print("validation and scoring ", time.time() - start)
        log_end(key=constants.BLOCK_STOP,
                     metadata={'first_epoch_num': first_epoch},
                     sync=False)

    train_meter.stop()
    status = 'success' if current_bleu >= tgt_bleu else 'aborted'
    log_end(key=constants.RUN_STOP,
                 metadata={'status': status})
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr, shuffling_seeds):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr()
    progress = progress_bar.build_progress_bar(args, itr, epoch_itr.epoch, no_progress_bar='simple')

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    if args.enable_parallel_backward_allred_opt and update_freq > 1:
        raise RuntimeError('--enable-parallel-backward-allred-opt is incompatible with --update-freq > 1')

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    if args.time_step :
        begin = time.time()
        end = time.time()
    count = 0

    #profile_count = 13
    profile_count = 10000000000

    for i, sample in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        if args.time_step :
            start_step = time.time()
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False, last_step=(i == len(itr)-1))
            continue
        else:
            log_output = trainer.train_step(sample, update_params=True, last_step=(i == len(itr)-1))
        if args.time_step :
            end_step = time.time()
            #if count > 10  and sample['target'].size(0) > 248 :
            seqs = sample['target'].size(0)
            srclen = sample['net_input']['src_tokens'].size(1)
            tgtlen = sample['target'].size(1)
            srcbatch = srclen * seqs
            tgtbatch = tgtlen * seqs
            #print("ITER {}> Seqs: {} SrcLen: {} TgtLen: {} Src Batch: {} Tgt Batch {}".format( count, seqs, srclen, tgtlen, srcbatch, tgtbatch))
            print("ITER {}> Seqs: {} SrcLen: {} TgtLen: {} Total Time: {:.3} Step Time: {:.3} Load Time: {:.3}".format( \
                count,                                                                                                  \
                sample['target'].size(0),                                                                               \
                sample['net_input']['src_tokens'].size(1),                                                              \
                sample['target'].size(1),                                                                               \
                (end_step-begin)*1000.0,                                                                                \
                (end_step-start_step)*1000.0,                                                                           \
                (start_step-end)*1000.0))
            count += 1
            begin = time.time()

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        if args.profile is not None and i == args.profile:
            import sys
            sys.exit()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
            valid_losses = validate(args, trainer, task, epoch_itr,
                                    [first_valid], shuffling_seeds)
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break
        if args.time_step :
            end = time.time()

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)

    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    return stats


def validate(args, trainer, task, epoch_itr, subsets, shuffling_seeds):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=trainer.get_model().max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seeds=shuffling_seeds,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            bucket_growth_factor=args.bucket_growth_factor,
            seq_len_multiple=args.seq_len_multiple,
            batching_scheme=args.batching_scheme,
            batch_multiple_strategy=args.batch_multiple_strategy,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    return valid_losses

def _get_ngrams_with_counter(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts

class RefBleuStats:
  def __init__(self, matches_by_order, possible_matches_by_order, reference_length, translation_length):
    self.matches_by_order = matches_by_order
    self.possible_matches_by_order = possible_matches_by_order
    self.reference_length = reference_length
    self.translation_length = translation_length

  def __add__(self, other):
    return RefBleuStats(
        [a+b for a,b in zip(self.matches_by_order, other.matches_by_order)],
        [a+b for a,b in zip(self.possible_matches_by_order, other.possible_matches_by_order)],
        self.reference_length + other.reference_length,
        self.translation_length + other.translation_length)

def compute_bleu(reference_corpus, translation_corpus, args, max_order=4, use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    args: CLI arguments
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
    translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

    overlap = dict((ngram, min(count, translation_ngram_counts[ngram])) for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]

  precisions = [0] * max_order
  smooth = 1.0

  # do reductions of matches_by_order and possible_matches_by_order
  if args.distributed_world_size > 1:
    stats = RefBleuStats(matches_by_order, possible_matches_by_order, reference_length, translation_length)
    all_stats = distributed_utils.all_gather_list(stats)
    stats = reduce(lambda a,b : a+b, all_stats)
    matches_by_order = stats.matches_by_order
    possible_matches_by_order = stats.possible_matches_by_order
    reference_length = stats.reference_length
    translation_length = stats.translation_length

  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    if reference_length > 0:
      ratio = translation_length / reference_length
      bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    else:
      bp = 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)*100.0

def detokenize_subtokenized_sentence(subtokenized_sentence):
  l1 = ' '.join(''.join(subtokenized_sentence.strip().split()).split('_'))
  l1 = l1.replace(' ,',',')
  l1 = l1.replace(' .','.')
  l1 = l1.replace(' !','!')
  l1 = l1.replace(' ?','?')
  l1 = l1.replace(' \' ','\'')
  l1 = l1.replace(' - ','-')
  l1 = l1.strip()
  return l1

class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols."""

    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
      return "".join(six.unichr(x) for x in range(sys.maxunicode)
                     if unicodedata.category(six.unichr(x)).startswith(prefix))

uregex = UnicodeRegex()

def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.
    See https://github.com/moses-smt/mosesdecoder/'
    'blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).
    Note that a numer (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    Args:
    string: the input string
    Returns:
    a list of tokens
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()

def score(args, trainer, task, epoch_itr, subset):

    log_start(key=constants.EVAL_START, metadata={'epoch_num': epoch_itr.epoch}, sync=False)
    begin = time.time()

    if not subset in task.datasets.keys():
        task.load_dataset(subset)

    src_dict = deepcopy(task.source_dictionary) # This is necessary, generation of translations
    tgt_dict = deepcopy(task.target_dictionary) # alters target dictionary messing up with the rest of training

    model = trainer.get_model()

    # Initialize data iterator
    itr = data.EpochBatchIterator(
        dataset=task.dataset(subset),
        max_tokens=min(2560,args.max_tokens),
        max_sentences=max(8,min((math.ceil(1024/args.distributed_world_size) // 4) * 4,128)),
        max_positions=(256,256),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        seq_len_multiple=args.seq_len_multiple,
        # Use a large growth factor to get fewer buckets.
        # Fewer buckets yield faster eval since batches are filled from single bucket
        # and eval dataset is small.
        bucket_growth_factor=1.2,
        batching_scheme=args.batching_scheme,
        batch_multiple_strategy=args.batch_multiple_strategy,
        ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    translator = SequenceGenerator(
	[model], tgt_dict, beam_size=args.beam,
	stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
	len_penalty=args.lenpen,
	sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        )
    # Generate and compute BLEU
    ref_toks = []
    sys_toks = []
    num_sentences = 0
    has_target = True
    if args.log_translations:
        log = open(os.path.join(args.save_dir, 'translations_epoch{}_{}'.format(epoch_itr.epoch, args.distributed_rank)), 'w+')
    with progress_bar.build_progress_bar(args, itr) as progress:
        translations = translator.generate_batched_itr(
                progress, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=True, timer=gen_timer, prefix_size=args.prefix_size,
                )

        wps_meter = TimeMeter()
        for sample_id, src_tokens, target_tokens, hypos in translations:
            # Process input and grount truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            src_str = src_dict.string(src_tokens, args.remove_bpe)
            if has_target:
                target_str = tgt_dict.string(target_tokens, args.remove_bpe)

            if args.log_translations:
                log.write('S-{}\t{}\n'.format(sample_id, src_str))
                if has_target:
                    log.write('T-{}\t{}\n'.format(sample_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict = None,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe
                        )
                if args.log_translations:
                    log.write('H-{}\t{}\t{}\n'.format(sample_id, hypo['score'], hypo_str))
                    # log.write(str(hypo_tokens))
                    log.write('P-{}\t{}\n'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        ))
                    ))

                # Score only the top hypothesis
                if has_target and i==0:
                    src_str = detokenize_subtokenized_sentence(src_str)
                    target_str = detokenize_subtokenized_sentence(target_str)
                    hypo_str = detokenize_subtokenized_sentence(hypo_str)
                    sys_tok = bleu_tokenize((hypo_str.lower() if args.ignore_case else hypo_str))
                    ref_tok = bleu_tokenize((target_str.lower() if args.ignore_case else target_str))
                    sys_toks.append(sys_tok)
                    ref_toks.append(ref_tok)

            wps_meter.update(src_tokens.size(0))
            progress.log({'wps':round(wps_meter.avg)})
            num_sentences += 1

    bleu_score_reference = compute_bleu(ref_toks, sys_toks, args)
    bleu_score_reference_str = '{:.4f}'.format(bleu_score_reference)
    if args.log_translations:
        log.close()
    if gen_timer.sum != 0:
        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1./gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: bleu_score={}'.format(subset, args.beam, bleu_score_reference_str))
    print('| Eval completed in: {:.2f}s'.format(time.time()-begin))
    log_end(key=constants.EVAL_STOP, metadata={'epoch_num': epoch_itr.epoch}, sync=False)

    return bleu_score_reference

def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
