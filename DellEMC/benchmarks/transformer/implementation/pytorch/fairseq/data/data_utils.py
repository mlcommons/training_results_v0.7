# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import itertools
import math
import os
import statistics
import time

import numpy as np
import torch

from . import FairseqDataset
import fairseq.data.batch_C_v0p5
import fairseq.data.batch_C_v0p5_better
import fairseq.data.batch_C_v0p6
import sys


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    print('Infer language pair from filename...')
    for filename in os.listdir(path):
        print('filename:', filename)
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


class ShardedIterator(object):
    """A sharded wrapper around an iterable (padded to length)."""

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count."""

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def skip(self, num_to_skip):
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False, n_seq_per_batch_multiple=8, seq_len_multiple=1):
    """ Convert a list of 1d tensors into a padded 2d tensor.

    Args:
        values: Python list where each element is a PyT 1d tensor
        pad_idx: The index into the translation dictionary for the pad token (typically refer to 'dict.pad()')
        eos_idx: The index into the translation dictionary for the eos token (typically refer to 'dict.eos()')
        left_pad: Bool, left- or right-padding (true: left, false: right)
        move_eos_to_beginning: Reverse order of sequence of tokens (true: reverse, false:leave in original order)
        n_seq_per_batch_multiple: The number of sequences per batch to round down to
        seq_len_multiple: The number of tokens per sequence to round up to
    """
    size_of_seq_dim = max(v.size(0) for v in values)  # Unpadded size
    n_seq_in_batch = len(values)

    if n_seq_per_batch_multiple % seq_len_multiple == 0:
        n_seq_multiple = n_seq_per_batch_multiple / seq_len_multiple
    else:
        n_seq_multiple = n_seq_per_batch_multiple

    if n_seq_in_batch < n_seq_multiple or n_seq_in_batch % n_seq_multiple > 0:
        seq_len_multiple = n_seq_per_batch_multiple

    size_of_seq_dim = (size_of_seq_dim + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple   # Padded seq len, rounded up to next multiple

    padded_2d_tensor = values[0].new(len(values), size_of_seq_dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()

        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if left_pad:
        for idx, val in enumerate(values):
            copy_tensor(val, padded_2d_tensor[idx][size_of_seq_dim - len(val):])
    else:
        for idx, val in enumerate(values):
            copy_tensor(val, padded_2d_tensor[idx][:len(val)])

    return padded_2d_tensor


class EpochBatchIterator(object):
    """Iterate over a FairseqDataset and yield batches bucketed by size.

    Batches may contain sequences of different lengths. This iterator can be
    reused across multiple epochs with the next_epoch_itr() method.

    Args:
        dataset: a FairseqDataset
        max_tokens: max number of tokens in each batch
        max_sentences: max number of sentences in each batch
        max_positions: max sentence length supported by the model
        ignore_invalid_inputs: don't raise Exception for sentences that are too long
        required_batch_size_multiple: require batch size to be a multiple of N
        seeds: seeds for random number generator for reproducibility (1 seed for
            each training epoch)
        num_shards: shard the data iterator into N shards
        shard_id: which shard of the data iterator to return
    """

    def __init__(
        self, dataset, dataloader_num_workers=1, dataloader_pin_memory=False, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1, seeds=[1],
        num_shards=1, shard_id=0, epoch=0, bucket_growth_factor=1.1, seq_len_multiple=1,
        batching_scheme='v0p5', batch_multiple_strategy='multiple_of_sequences',
    ):
        assert isinstance(dataset, FairseqDataset)
        self.dataset = dataset
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')

        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory

        assert len(max_positions) == 2, "Max positions contains source and target lengths!"
        max_src_pos,max_tgt_pos = max_positions
        self.max_positions = max_positions
        self.max_positions_num = min(max_src_pos, max_tgt_pos)

        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.bsz_mult = required_batch_size_multiple
        self.seeds = seeds
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.seq_len_multiple = seq_len_multiple
        self.batching_scheme = batching_scheme
        self.batch_multiple_strategy = batch_multiple_strategy

        self.epoch = epoch
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

        with numpy_seed(self.seeds[0]):
            import time
            start = time.time()
            indices = self.dataset.ordered_indices(self.seeds[self.epoch])
#need integer, rather than float('Inf') values
            max_sentences = max_sentences if max_sentences is not None else sys.maxsize
            max_tokens = max_tokens if max_tokens is not None else sys.maxsize

            if self.batching_scheme == 'v0p5' :
                batches = fairseq.data.batch_C_v0p5.make_batches_v0p5(self.dataset.src_sizes, self.dataset.tgt_sizes, indices, max_tokens, max_sentences, self.bsz_mult, self.max_positions_num)
            elif self.batching_scheme == 'v0p5_better' :
                print('self.dataset.src_sizes', self.dataset.src_sizes.size)
                print('self.dataset.tgt_sizes', self.dataset.tgt_sizes.size)
                batches = fairseq.data.batch_C_v0p5_better.make_batches_v0p5_better(self.dataset.src_sizes, self.dataset.tgt_sizes, indices, max_tokens, max_sentences, self.max_positions_num, self.bsz_mult, self.seq_len_multiple)
            elif self.batching_scheme == 'v0p6':
                batch_strategy = 2
                if self.batch_multiple_strategy == 'mult_of_sequences':
                    batch_strategy = 0
                elif self.batch_multiple_strategy == 'pad_sequence_to_mult':
                    batch_strategy = 1
                elif self.batch_multiple_strategy == 'dynamic':
                    batch_strategy = 2
                else:
                    assert False, "Unknown batch multiple strategy!"

                bucket_specify_min_boundary = 8
                use_efficient_last_pack = False
                #batch_strategy = 2
                batches = fairseq.data.batch_C_v0p6.make_batches_v0p6(self.dataset.src_sizes,
                                                                      self.dataset.tgt_sizes,
                                                                      indices,
                                                                      max_tokens,
                                                                      max_sentences,
                                                                      self.bsz_mult,
                                                                      self.max_positions_num,
                                                                      bucket_specify_min_boundary,
                                                                      bucket_growth_factor,
                                                                      batch_strategy,
                                                                      use_efficient_last_pack)
            else : # reference
                def roundup(x, multiple):
                    return (x + multiple - 1) // multiple * multiple

                def rounddown(x, multiple):
                    return x // multiple * multiple

                def create_bucket_bounds_lists(max_allowable_seq_length, bucket_specify_min_boundary, bucket_specify_growth_scale):
                    bucket_boundaries = []
                    x = bucket_specify_min_boundary
                    while x < max_allowable_seq_length:
                        bucket_boundaries.append(x)
                        x = max(x + 1, int(x * bucket_specify_growth_scale))

                    if use_efficient_last_pack:
                        buckets_min_list = [0] + [i+1 for i in bucket_boundaries]
                        buckets_max_list = bucket_boundaries + [max_allowable_seq_length]
                    else:
                        buckets_min_list = [0] + bucket_boundaries
                        buckets_max_list = bucket_boundaries + [max_allowable_seq_length + 1]

                    return buckets_min_list, buckets_max_list

                def create_seq_to_bucket_id_list_and_n_seq_per_batch(n_tok_per_seq, max_allowable_seq_length, max_sentences, pad_seq_per_batch_to_multiple_of, pad_tok_per_seq_to_multiple_of, bucket_specify_min_boundary, bucket_specify_growth_scale):
                    bucket_interval_min, bucket_interval_max = create_bucket_bounds_lists(max_allowable_seq_length, bucket_specify_min_boundary, bucket_specify_growth_scale)

                    if do_seq_len_padding_to_multiple:
                        n_seq_per_batch = [max_tokens // roundup(x, pad_tok_per_seq_to_multiple_of) for x in bucket_interval_max]
                    elif do_batch_size_rounding_down_to_multiple:
                        n_seq_per_batch = [rounddown(max_tokens // x, pad_seq_per_batch_to_multiple_of) for x in bucket_interval_max]
                    elif do_dynamic_batch_size_choice:
                        n_seq_per_batch_based_on_seq_len = [max_tokens // roundup(x, pad_tok_per_seq_to_multiple_of) for x in bucket_interval_max]
                        n_seq_per_batch_based_on_n_seq = [rounddown(max_tokens // x, pad_seq_per_batch_to_multiple_of) for x in bucket_interval_max]
                        n_seq_per_batch = [max(a,b) for a, b in zip(n_seq_per_batch_based_on_seq_len, n_seq_per_batch_based_on_n_seq)]
                    else:
                        n_seq_per_batch = [max_tokens // x for x in bucket_interval_max]
                    n_seq_per_batch = [min(max_sentences, i) if max_sentences is not None else i for i in n_seq_per_batch]

                    for a, b, c in zip(bucket_interval_min, bucket_interval_max, n_seq_per_batch):
                        print('bucket:', a, b, c)

                    token_length_2_bucket_id = {}
                    for x in range(max_allowable_seq_length+1):
                        for bucket_id, payload in enumerate(zip(bucket_interval_min, bucket_interval_max)):
                            bmin, bmax = payload
                            if (bmin <= x and x <= bmax and use_efficient_last_pack) or (bmin <= x and x < bmax):
                                token_length_2_bucket_id[x] = bucket_id
                                break

                    return ([token_length_2_bucket_id[x] if x <= max_allowable_seq_length else -1 for x in n_tok_per_seq], n_seq_per_batch, len(bucket_interval_min))


                # Make adjustments to tuneable parameters here
                pad_seq_per_batch_to_multiple_of = self.bsz_mult
                pad_tok_per_seq_to_multiple_of = self.bsz_mult
                max_allowable_seq_length = self.max_positions_num
                bucket_specify_min_boundary = 8
                bucket_specify_growth_scale = bucket_growth_factor  ##1.035
                do_seq_len_padding_to_multiple = False
                do_batch_size_rounding_down_to_multiple = False
                do_dynamic_batch_size_choice = True
                use_efficient_last_pack = False

                batches = []
                src_token_counts = []
                dst_token_counts = []
                seq_counts = []
                padded_token_counts = []
                batch_max_padded_seq_len = 0
                batch_seq_count = 0
                batches.append([])
                src_batch_token_count = 0
                dst_batch_token_count = 0
                curr_batch_padded_token_count = 0
                batch_n_seq = 0
                bucket_id = 0
                longest_in_batch = []

                print('### max_tokens:', max_tokens)
                print('### max_sentences:', max_sentences)

                pairwise_max_seq_len = [max(a,b) for a, b in zip(dataset.src_sizes, dataset.tgt_sizes)]
                bucket_ids, n_seq_per_batch, n_buckets = create_seq_to_bucket_id_list_and_n_seq_per_batch(pairwise_max_seq_len, max_allowable_seq_length, max_sentences, pad_seq_per_batch_to_multiple_of, pad_tok_per_seq_to_multiple_of, bucket_specify_min_boundary, bucket_specify_growth_scale)

                buckets = []
                for i in range(n_buckets):
                    buckets.append([])

                n_rejected_sequences = 0
                for idx, bidx in enumerate(bucket_ids):
                    if bidx >= 0:
                        buckets[bidx].append(idx)
                    else:
                        n_rejected_sequences += 1

                # Remove empty buckets (causes blow-up in eval code).
                buckets = [i for i in buckets if len(i) > 0]

                print(n_rejected_sequences, 'were omitted due to containing over 256 tokens.')

                batch_seq_count = 0
                #count = 0
                seq_len_tracker = 0
                for bucket, nspb in zip(buckets, n_seq_per_batch):
                    for item in bucket:
                        if batch_n_seq < nspb:
                            batches[-1].append(item)
                            src_batch_token_count += dataset.src_sizes[item]
                            dst_batch_token_count += dataset.tgt_sizes[item]
                            seq_len_tracker = max(seq_len_tracker, dst_batch_token_count)
                            batch_n_seq += 1
                        else:
                            batches.append([item])

                            src_token_counts.append(src_batch_token_count)
                            dst_token_counts.append(dst_batch_token_count)

                            src_batch_token_count = dataset.src_sizes[item]
                            dst_batch_token_count = dataset.tgt_sizes[item]

                            seq_counts.append(batch_n_seq)
                            batch_n_seq = 1

                    batches.append([])
                    batch_n_seq = 0
                    seq_counts.append(batch_n_seq)
                    src_batch_token_count = 0
                    dst_batch_token_count = 0
                    src_token_counts.append(src_batch_token_count)
                    dst_token_counts.append(dst_batch_token_count)


                seq_cnt2 = []
                for batch in batches:
                    seq_len_tracker = 0
                    nseqbucket = 0
                    for item in batch:
                        a = dataset.src_sizes[item]
                        b = dataset.tgt_sizes[item]
                        seq_len_tracker = max(seq_len_tracker, max(a, b))
                        nseqbucket += 1

                    longest_in_batch.append(seq_len_tracker)
                    seq_cnt2.append(nseqbucket)

                # In the unlucky case, remove a newly created but empty last batch
                if not batches[-1]:
                    del batches[-1]
                    del seq_counts[-1]
                    del src_token_counts[-1]
                    del dst_token_counts[-1]

                tmp_batches = batches
                batches = []
                for b in tmp_batches:
                    if b:
                        batches.append(b)

                #padded_token_counts = src_token_counts
                #padded_token_counts = [x*0 for x in src_token_counts]   # Setting to zero until this is actually implemented
                #print('split dataset length:', len(dataset.src))
                #print('mean src tokens per batch =', statistics.mean(src_token_counts), statistics.mean(padded_token_counts))
                #print('median src tokens per batch =', statistics.median(src_token_counts), statistics.median(padded_token_counts))
                #print('stdev src tokens per batch =', statistics.stdev(src_token_counts), statistics.stdev(padded_token_counts))
                #print('min src tokens per batch =', min(src_token_counts), min(padded_token_counts))
                #print('max src tokens per batch =', max(src_token_counts), max(padded_token_counts))

                #print('mean tgt tokens per batch =', statistics.mean(dst_token_counts), statistics.mean(padded_token_counts))
                #print('median tgt tokens per batch =', statistics.median(dst_token_counts), statistics.mean(padded_token_counts))
                #print('stdev tgt tokens per batch =', statistics.stdev(dst_token_counts), statistics.stdev(padded_token_counts))
                #print('min tgt tokens per batch =', min(dst_token_counts), min(padded_token_counts))
                #print('max tgt tokens per batch =', max(dst_token_counts), max(padded_token_counts))

                #print('mean seq per batch =', statistics.mean(seq_counts), statistics.mean(padded_token_counts))
                #print('median seq per batch =', statistics.median(seq_counts), statistics.median(padded_token_counts))
                #print('stdev seq per batch =', statistics.stdev(seq_counts), statistics.stdev(padded_token_counts))
                #print('min seq per batch =', min(seq_counts), min(padded_token_counts))
                #print('max seq per batch =', max(seq_counts), max(padded_token_counts))

                #print('pad inc: mean tgt tokens per batch =', statistics.mean(np.array(seq_cnt2) * np.array(longest_in_batch)), longest_in_batch[:3], seq_cnt2[:3])
                #print('pad inc: median tgt tokens per batch =', statistics.median(np.array(seq_cnt2) * np.array(longest_in_batch)), longest_in_batch[:3], seq_cnt2[:3])

            self.frozen_batches = tuple(batches)
#            self.frozen_batches = tuple(self._batch_generator())
            print("generated %d batches in %fs" % (len(batches), time.time() - start))

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True):
        """Shuffle batches and return a new iterator over the dataset."""
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(self.epoch, shuffle)
        return self._cur_epoch_itr

    def end_of_epoch(self):
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            if itr_pos < len(itr):
                self._next_epoch_itr = itr.skip(itr_pos)

    def _get_iterator_for_epoch(self, epoch, shuffle):
        if shuffle:
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with numpy_seed(self.seeds[epoch]):
                batches = list(self.frozen_batches)  # copy
                np.random.shuffle(batches)
        else:
            batches = self.frozen_batches

        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collater,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            batch_sampler=ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[]),
        ))

    def _batch_generator(self):
        batch = []

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False

            if len(batch) == self.max_sentences:
                return True

            if num_tokens > self.max_tokens:
                return True

            return False


        sample_len = 0
        sample_lens = []
        ignored = []
        for idx in self.dataset.ordered_indices(self.seeds[self.epoch]):
            if not self.dataset.valid_size(idx, self.max_positions):
                if self.ignore_invalid_inputs:
                    ignored.append(idx)
                    continue

                raise Exception((
                'Size of sample #{} is invalid, max_positions={}, skip this example with --skip-invalid-size-inputs-valid-test'
                ).format(idx, self.max_positions))

            sample_lens.append(self.dataset.num_tokens(idx))
            sample_len = max(sample_len, sample_lens[-1])
            num_tokens = (len(batch) + 1) * sample_len

            if is_batch_full(num_tokens):
                mod_len = max(self.bsz_mult * (len(batch) // self.bsz_mult), len(batch) % self.bsz_mult,)
                yield batch[:mod_len]
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

            batch.append(idx)

        if len(batch) > 0:
            yield batch

        if len(ignored) > 0:
            print((
            '| WARNING: {} samples have invalid sizes and will be skipped, max_positions={}, first few sample ids={}'
            ).format(len(ignored), self.max_positions, ignored[:10]))


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and restores the state afterward"""
    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)
