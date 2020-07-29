#!/usr/bin/env python
import argparse
import logging
import os
from ast import literal_eval

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from mlperf_logging.mllog import constants

import seq2seq.data.config as config
import seq2seq.train.trainer as trainers
import seq2seq.utils as utils
from seq2seq.data.dataset import LazyParallelDataset, PreprocessedDataset
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.inference import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.utils import log_start, log_end, log_event, configure_logger
from seq2seq.train.lr_scheduler import WarmupMultiStepLR


def parse_args():
    """
    Parse commandline arguments.
    """
    def exclusive_group(group, name, default, help):
        destname = name.replace('-', '_')
        subgroup = group.add_mutually_exclusive_group(required=False)
        subgroup.add_argument(f'--{name}', dest=f'{destname}',
                              action='store_true',
                              help=f'{help} (use \'--no-{name}\' to disable)')
        subgroup.add_argument(f'--no-{name}', dest=f'{destname}',
                              action='store_false', help=argparse.SUPPRESS)
        subgroup.set_defaults(**{destname: default})

    parser = argparse.ArgumentParser(
        description='GNMT training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en',
                         help='path to the directory with training/test data')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset')
    dataset.add_argument('--preproc-data-dir', default='/tmp/preprocessed',
                         help='path to the directory with preprocessed \
                         training/test data')
    exclusive_group(group=dataset, name='use-preproc-data', default=True,
                    help='use preprocessed dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it will be \
                         automatically created if it does not exist')
    results.add_argument('--save', default='gnmt',
                         help='defines subdirectory within RESULTS_DIR for \
                         results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--hidden-size', default=1024, type=int,
                       help='model hidden size')
    model.add_argument('--num-layers', default=4, type=int,
                       help='number of RNN layers in encoder and in decoder')
    model.add_argument('--dropout', default=0.2, type=float,
                       help='dropout applied to input of RNN cells')

    exclusive_group(group=model, name='share-embedding', default=True,
                    help='use shared embeddings for encoder and decoder')

    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                       CrossEntropyLoss, if not zero model will be trained \
                       with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16',
                         choices=['fp32', 'fp16'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='master seed for random number generators, if \
                         "seed" is undefined then the master seed will be \
                         sampled from random.SystemRandom()')
    general.add_argument('--prealloc-mode', default='always', type=str,
                         choices=['off', 'once', 'always'],
                         help='controls preallocation')

    exclusive_group(group=general, name='eval', default=True,
                    help='run validation and test after every epoch')
    exclusive_group(group=general, name='env', default=False,
                    help='print info about execution env')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')
    exclusive_group(group=general, name='log-all-ranks', default=True,
                    help='enables logging from all distributed ranks, if \
                    disabled then only logs from rank 0 are reported')
    exclusive_group(group=general, name='fused-attention', default=False,
                    help='enables fused attention')
    exclusive_group(group=general, name='fused-xentropy', default=True,
                    help='enables fused cross cross entropy with label \
                    smoothing')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--train-batch-size', default=128, type=int,
                          help='training batch size per worker')
    training.add_argument('--train-global-batch-size', default=None, type=int,
                          help='global training batch size, this argument \
                          does not have to be defined, if it is defined it \
                          will be used to automatically \
                          compute train_iter_size \
                          using the equation: train_iter_size = \
                          train_global_batch_size // (train_batch_size * \
                          world_size)')
    training.add_argument('--train-iter-size', metavar='N', default=1,
                          type=int,
                          help='training iter size, training loop will \
                          accumulate gradients over N iterations and execute \
                          optimizer every N steps')
    training.add_argument('--epochs', default=8, type=int,
                          help='max number of training epochs')

    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enables gradient clipping and sets maximum \
                          norm of gradients')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--train-loader-workers', default=1, type=int,
                          help='number of workers for training data loading')
    training.add_argument('--batching', default='bucketing', type=str,
                          choices=['random', 'sharding', 'bucketing'],
                          help='select batching algorithm')
    training.add_argument('--shard-size', default=80, type=int,
                          help='shard size for "sharding" batching algorithm, \
                          in multiples of global batch size')
    training.add_argument('--num-buckets', default=5, type=int,
                          help='number of buckets for "bucketing" batching \
                          algorithm')

    # optimizer
    optimizer = parser.add_argument_group('optimizer setup')
    optimizer.add_argument('--optimizer', type=str, default='Adam',
                           help='training optimizer')
    optimizer.add_argument('--lr', type=float, default=1.00e-3,
                           help='learning rate')
    optimizer.add_argument('--optimizer-extra', type=str,
                           default="{}",
                           help='extra options for the optimizer')

    # mixed precision loss scaling
    loss_scaling = parser.add_argument_group('mixed precision loss scaling \
                                             setup')
    loss_scaling.add_argument('--init-scale', type=float, default=1024,
                              help='initial loss scale')
    loss_scaling.add_argument('--upscale-interval', type=float, default=128,
                              help='loss upscaling interval')

    # scheduler
    scheduler = parser.add_argument_group('learning rate scheduler setup')
    scheduler.add_argument('--warmup-steps', type=str, default='200',
                           help='number of learning rate warmup iterations')
    scheduler.add_argument('--remain-steps', type=str, default='0.666',
                           help='starting iteration for learning rate decay')
    scheduler.add_argument('--decay-interval', type=str, default='None',
                           help='interval between learning rate decay steps')
    scheduler.add_argument('--decay-steps', type=int, default=4,
                           help='max number of learning rate decay steps')
    scheduler.add_argument('--decay-factor', type=float, default=0.5,
                           help='learning rate decay factor')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--max-length-test', default=150, type=int,
                      help='maximum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--min-length-test', default=0, type=int,
                      help='minimum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--intra-epoch-eval', metavar='N', default=0, type=int,
                      help='evaluate within training epoch, this option will \
                      enable extra N equally spaced evaluations executed \
                      during each training epoch')
    test.add_argument('--test-loader-workers', default=0, type=int,
                      help='number of workers for test data loading')

    # checkpointing
    chkpt = parser.add_argument_group('checkpointing setup')
    chkpt.add_argument('--start-epoch', default=0, type=int,
                       help='manually set initial epoch counter')
    chkpt.add_argument('--resume', default=None, type=str, metavar='PATH',
                       help='resumes training from checkpoint from PATH')
    chkpt.add_argument('--save-all', action='store_true', default=False,
                       help='saves checkpoint after every epoch')
    chkpt.add_argument('--save-freq', default=5000, type=int,
                       help='save checkpoint every SAVE_FREQ batches')
    chkpt.add_argument('--keep-checkpoints', default=0, type=int,
                       help='keep only last KEEP_CHECKPOINTS checkpoints, \
                       affects only checkpoints controlled by --save-freq \
                       option')

    # benchmarking
    benchmark = parser.add_argument_group('benchmark setup')
    benchmark.add_argument('--target-bleu', default=24.0, type=float,
                           help='target accuracy, training will be stopped \
                           when the target is achieved')

    # distributed
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='global rank of the process, do not set!')
    distributed.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                             help='local rank of the process, do not set!')
    distributed.add_argument('--enable-apex-allreduce-overlap',
                             action='store_true', default=False,
                             help='enable overlap of allreduce communication \
                             with bprop')
    distributed.add_argument('--apex-num-allreduce-streams',
                             default=1, type=int,
                             help='num. allreduce streams')
    distributed.add_argument('--apex-message-size', default=1e7, type=int,
                             help='min. number of elements in communication \
                             bucket')

    # distributed weight update
    dwu_group = parser.add_argument_group('distributed weight update setup')
    dwu_group.add_argument('--distributed-weight-update', '--dwu', default=0, type=int, metavar='DWU',
                       help='select distributed weight update strategy')
    dwu_group.add_argument('--dwu-group-size', '--dwugs', default=0, type=int, metavar='DWUGS',
                       help='distributed weight update group size. If arg is 0, defaults to one node')
    dwu_group.add_argument('--dwu-num-blocks', '--dwunb', default=8, type=int, metavar='DWUNB',
                       help='number of blocks in dwu scheme')
    dwu_group.add_argument('--dwu-num-chunks', '--dwuchks', default=4, type=int,
                       help='number of chunks of each parameters block')
    dwu_group.add_argument('--dwu-num-rs-pg', '--dwurspg', default=2, type=int, metavar='DWURSPG',
                       help='number of reduction-scatter streams in dwu scheme')
    dwu_group.add_argument('--dwu-num-ar-pg', '--dwuarpg', default=4, type=int, metavar='DWUARPG',
                       help='number of all-reduce streams in dwu scheme')
    dwu_group.add_argument('--dwu-num-ag-pg', '--dwuagpg', default=2, type=int, metavar='DWUAGPG',
                       help='number of all-gather streams in dwu scheme')
    dwu_group.add_argument('--dwu-full-pipeline', action='store_true', 
                       help='whether to do full or partial pipeline')
    dwu_group.add_argument('--dwu-overlap-reductions', action='store_true',
                       help='whether to overlap reductions with backprop')
    dwu_group.add_argument('--dwu-grad-norm', action='store_true',
                       help='whether to compute L2 grad norm')
    dwu_group.add_argument('--dwu-e5m2-allgather', action='store_true',
                       help='whether to use e5m2 allgather')

    args = parser.parse_args()

    args.warmup_steps = literal_eval(args.warmup_steps)
    args.remain_steps = literal_eval(args.remain_steps)
    args.decay_interval = literal_eval(args.decay_interval)

    return args


def build_criterion(vocab_size, padding_idx, smoothing, fusion):
    if smoothing == 0.:
        logging.info(f'Building CrossEntropyLoss')
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    else:
        logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing, fusion=fusion)
    return criterion


def main():
    """
    Launches data-parallel multi-gpu training.
    """

    configure_logger(constants.GNMT)
    log_start(key=constants.INIT_START, log_all_ranks=True)

    args = parse_args()
    device = utils.set_device(args.cuda, args.local_rank)
    distributed = utils.init_distributed(args.cuda)

    # preinit and warmup streams/ groups for apex DDP communicators
    # distributed weight update doesn't require this
    allreduce_communicators=None
    if distributed and args.distributed_weight_update == 0 and \
            args.apex_num_allreduce_streams > 1:
        bucket_pgs = [torch.distributed.new_group() for _ in range(args.apex_num_allreduce_streams)]
        bucket_streams = [torch.cuda.Stream() for _ in range(args.apex_num_allreduce_streams)]
        for pg, stream in zip(bucket_pgs,bucket_streams):
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(torch.cuda.FloatTensor(1), group=pg)
        allreduce_communicators=(bucket_pgs,bucket_streams)

    args.rank = utils.get_rank()

    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_rank_{utils.get_rank()}.log'
    utils.setup_logging(args.log_all_ranks,
                        os.path.join(save_path, log_filename))

    if args.env:
        utils.log_env_info()

    logging.info(f'Saving results to: {save_path}')
    logging.info(f'Run arguments: {args}')

    # additional argument check
    if args.math == 'fp32' and (args.fused_attention or args.fused_xentropy):
        logging.warn(f'Only support FP16 `--fused-attention` and '
            '`--fused-xentropy`, disabling them')
        args.fused_attention = args.fused_xentropy = False

    # automatically set train_iter_size based on train_global_batch_size,
    # world_size and per-worker train_batch_size
    if args.train_global_batch_size is not None:
        global_bs = args.train_global_batch_size
        bs = args.train_batch_size
        world_size = utils.get_world_size()
        assert global_bs % (bs * world_size) == 0
        args.train_iter_size = global_bs // (bs * world_size)
        logging.info(f'Global batch size was set in the config, '
                     f'Setting train_iter_size to {args.train_iter_size}')
    # setup L2 promotion
    if args.cuda:
        utils.l2_promote()

    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.epochs,
                                                      device)
    worker_seed = worker_seeds[args.rank]
    logging.info(f'Worker {args.rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)

    # build tokenizer
    # https://github.com/mlperf/policies/issues/201
    pad_vocab = utils.pad_vocabulary(args.math)
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME),
                          pad_vocab)

    vocab_size = tokenizer.vocab_size

    # build GNMT model
    model_config = {'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout, 'batch_first': False,
                    'share_embedding': args.share_embedding,
                    'fusion': args.fused_attention}
    model = GNMT(vocab_size=vocab_size, **model_config)
    logging.info(model)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing,
                                args.fused_xentropy)

    opt_config = {'optimizer': args.optimizer, 'lr': args.lr}
    opt_config.update(literal_eval(args.optimizer_extra))
    logging.info(f'Training optimizer config: {opt_config}')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    # create trainer
    save_info = {'model_config': model_config, 'config': args, 'tokenizer':
                 tokenizer.get_state()}
    loss_scaling = {'init_scale': args.init_scale, 'upscale_interval':
                    args.upscale_interval}
    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        iter_size=args.train_iter_size,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info=save_info,
        opt_config=opt_config,
        batch_first=model.batch_first,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        loss_scaling=loss_scaling,
        print_freq=args.print_freq,
        cuda=args.cuda,
        distributed=distributed,
        distributed_overlap_allreduce=args.enable_apex_allreduce_overlap,
        distributed_overlap_num_allreduce_streams=args.apex_num_allreduce_streams,
        distributed_overlap_allreduce_messagesize=args.apex_message_size,
        distributed_overlap_allreduce_communicators=allreduce_communicators,
        intra_epoch_eval=args.intra_epoch_eval,
        prealloc_mode=args.prealloc_mode)

    trainer_options['model'] = model
    trainer = trainers.Seq2SeqTrainer(args=args, **trainer_options)

    trainer.preallocate(args.train_batch_size, args.max_length_train,
                        training=True)

    log_end(key=constants.INIT_STOP, sync=False)
    log_start(key=constants.RUN_START, sync=True)
    utils.barrier()

    log_event(key=constants.MAX_SEQUENCE_LENGTH,
              value=args.max_length_train,
              metadata={'method': 'discard'})

    if args.use_preproc_data:
        train_data = PreprocessedDataset(
            min_len=args.min_length_train,
            max_len=args.max_length_train,
            vocab_size=tokenizer.vocab_size,
            )
        train_data.read_data(
            os.path.join(args.preproc_data_dir, 'training.bin'),
            tokenizer.vocab_size,
            )
        train_data.prepare()
    else:
        train_data = LazyParallelDataset(
            src_fname=os.path.join(args.dataset_dir, config.SRC_TRAIN_FNAME),
            tgt_fname=os.path.join(args.dataset_dir, config.TGT_TRAIN_FNAME),
            tokenizer=tokenizer,
            min_len=args.min_length_train,
            max_len=args.max_length_train,
            sort=False,
            max_size=args.max_size,
            )

    test_data = TextDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_test,
        max_len=args.max_length_test,
        sort=True)

    batching_opt = {'shard_size': args.shard_size,
                    'num_buckets': args.num_buckets}

    # get data loaders
    train_loader = train_data.get_loader(batch_size=args.train_batch_size,
                                         seeds=shuffling_seeds,
                                         batch_first=model.batch_first,
                                         shuffle=True,
                                         batching=args.batching,
                                         batching_opt=batching_opt,
                                         num_workers=args.train_loader_workers)

    log_event(key=constants.GLOBAL_BATCH_SIZE,
              value=args.train_batch_size * utils.get_world_size(),
              sync=False)

    test_loader = test_data.get_loader(batch_size=args.test_batch_size,
                                       batch_first=model.batch_first,
                                       shuffle=False,
                                       num_workers=args.test_loader_workers)

    log_event(key=constants.TRAIN_SAMPLES,
              value=train_loader.sampler.num_samples, sync=False)
    log_event(key=constants.EVAL_SAMPLES,
              value=len(test_loader.dataset), sync=False)

    translator = Translator(model=model,
                            tokenizer=tokenizer,
                            loader=test_loader,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_length_test,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.cuda,
                            print_freq=args.print_freq,
                            dataset_dir=args.dataset_dir,
                            target_bleu=args.target_bleu,
                            save_path=args.save_path)

    total_train_iters = len(train_loader) // args.train_iter_size * args.epochs

    scheduler_config = {'warmup_steps': args.warmup_steps,
                        'remain_steps': args.remain_steps,
                        'decay_interval': args.decay_interval,
                        'decay_steps': args.decay_steps,
                        'decay_factor': args.decay_factor}

    logging.info(f'Training LR schedule config: {scheduler_config}')
    scheduler = WarmupMultiStepLR(trainer.optimizer, total_train_iters,
                                  **scheduler_config)
    trainer.scheduler = scheduler
    trainer.translator = translator

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error(f'No checkpoint found at {args.resume}')

    # training loop
    break_training = False
    test_bleu = None
    for epoch in range(args.start_epoch, args.epochs):
        log_start(key=constants.BLOCK_START,
                  metadata={'first_epoch_num': epoch + 1,
                            'epoch_count': 1},
                  sync=False)
        log_start(key=constants.EPOCH_START,
                  metadata={'epoch_num': epoch + 1},
                  sync=False)

        logging.info(f'Starting epoch {epoch}')
        train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss, train_perf = trainer.optimize(train_loader)

        log_end(key=constants.EPOCH_STOP,
                metadata={'epoch_num': epoch + 1},
                sync=False)

        if args.eval:
            log_start(key=constants.EVAL_START,
                      metadata={'epoch_num': epoch + 1},
                      sync=False)
            test_bleu, break_training = translator.run(calc_bleu=True,
                                                       epoch=epoch)
            log_event(key=constants.EVAL_ACCURACY,
                      value=test_bleu / 100,
                      metadata={'epoch_num': epoch + 1},
                      sync=False)
            log_end(key=constants.EVAL_STOP,
                    metadata={'epoch_num': epoch + 1},
                    sync=False)

        acc_log = []
        acc_log += [f'Summary: Epoch: {epoch}']
        acc_log += [f'Training Loss: {train_loss:.4f}']
        if args.eval:
            acc_log += [f'Test BLEU: {test_bleu:.2f}']

        perf_log = []
        perf_log += [f'Performance: Epoch: {epoch}']
        perf_log += [f'Training: {train_perf:.0f} Tok/s']

        if args.rank == 0:
            logging.info('\t'.join(acc_log))
            logging.info('\t'.join(perf_log))

        logging.info(f'Finished epoch {epoch}')
        log_end(key=constants.BLOCK_STOP,
                metadata={'first_epoch_num': epoch + 1},
                sync=False)

        if break_training:
            break

    if args.use_preproc_data:
        train_data.finalize()

    status = 'success' if break_training else 'aborted'
    log_end(key=constants.RUN_STOP,
            metadata={'status': status},
            sync=False)


if __name__ == '__main__':
    main()
