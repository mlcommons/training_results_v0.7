import logging
import time
import os
import argparse

import torch
from torch.utils.data import DataLoader

from seq2seq.data.tokenizer import Tokenizer
import seq2seq.data.config as config
import seq2seq.utils as utils
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.dataset import PreprocessedDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='GNMT prepare data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset-dir', default='data/wmt16_de_en',
                        help='path to the directory with training/test data')
    parser.add_argument('--preproc-data-dir', default='/tmp/preprocessed',
                        help='path to the directory with preprocessed \
                        training/test data')
    parser.add_argument('--max-size', default=None, type=int,
                        help='use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset')

    parser.add_argument('--math', default='fp16',
                        choices=['fp32', 'fp16'],
                        help='arithmetic type')

    parser.add_argument('--max-length-train', default=50, type=int,
                        help='maximum sequence length for training \
                        (including special BOS and EOS tokens)')
    parser.add_argument('--min-length-train', default=0, type=int,
                        help='minimum sequence length for training \
                        (including special BOS and EOS tokens)')

    parser.add_argument('--rank', default=0, type=int,
                        help='global rank of the process, do not set!')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='local rank of the process, do not set!')

    args = parser.parse_args()
    return args


def build_collate_fn(max_seq_len, parallel=True):
    def collate_seq(seq):
        lengths = torch.tensor([len(s) for s in seq])
        batch_length = max_seq_len

        shape = (len(seq), batch_length)
        seq_tensor = torch.full(shape, config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[i, :end_seq].copy_(s[:end_seq])

        return (seq_tensor, lengths)

    def parallel_collate(seqs):
        src_seqs, tgt_seqs = zip(*seqs)
        return tuple([collate_seq(s) for s in [src_seqs, tgt_seqs]])

    return parallel_collate


def load_dataset(tokenizer, args):
    train_data = LazyParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TRAIN_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TRAIN_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=args.max_size)

    collate_fn = build_collate_fn(max_seq_len=args.max_length_train,
                                  parallel=True)

    loader = DataLoader(train_data,
                        batch_size=1024,
                        collate_fn=collate_fn,
                        num_workers=min(os.cpu_count(), 16),
                        timeout=120,
                        drop_last=False)

    srcs = []
    tgts = []
    src_lengths = []
    tgt_lengths = []

    for (src, src_len), (tgt, tgt_len) in loader:
        src_lengths.append(src_len)
        tgt_lengths.append(tgt_len)
        srcs.append(src)
        tgts.append(tgt)

    srcs = torch.cat(srcs)
    tgts = torch.cat(tgts)
    src_lengths = torch.cat(src_lengths)
    tgt_lengths = torch.cat(tgt_lengths)

    return srcs, tgts, src_lengths, tgt_lengths


def broadcast_dataset(world_size, rank, max_length_train, srcs, tgts,
        src_lengths, tgt_lengths):
    assert world_size > 1

    # Broadcast preprocessed dataset length
    if rank == 0:
        sizes = torch.tensor(src_lengths.shape, device='cuda',
            dtype=torch.int64)
    else:
        sizes = torch.zeros((1,), device='cuda', dtype=torch.int64)

    torch.distributed.broadcast(sizes, 0)
    nsamples = sizes.item()

    # Prepare tensor for receving preprocessed dataset
    if rank == 0:
        srcs_cuda, tgts_cuda, src_lengths_cuda, tgt_lengths_cuda = \
            srcs.cuda(), tgts.cuda(), src_lengths.cuda(), tgt_lengths.cuda()
    else:
        srcs_cuda = torch.empty((nsamples, max_length_train),
            device='cuda', dtype=torch.int64)
        tgts_cuda = torch.empty((nsamples, max_length_train),
            device='cuda', dtype=torch.int64)
        src_lengths_cuda = torch.empty((nsamples,), device='cuda',
            dtype=torch.int64)
        tgt_lengths_cuda = torch.empty((nsamples,), device='cuda',
            dtype=torch.int64)

    # Broadcast preprocessed dataset
    torch.distributed.broadcast(srcs_cuda, 0)
    torch.distributed.broadcast(tgts_cuda, 0)
    torch.distributed.broadcast(src_lengths_cuda, 0)
    torch.distributed.broadcast(tgt_lengths_cuda, 0)

    if rank > 0:
        srcs, tgts, src_lengths, tgt_lengths = srcs_cuda.cpu(), \
            tgts_cuda.cpu(), src_lengths_cuda.cpu(), tgt_lengths_cuda.cpu()

    return srcs, tgts, src_lengths, tgt_lengths


def main():
    args = parse_args()

    use_cuda = True
    device = utils.set_device(use_cuda, args.local_rank)
    distributed = utils.init_distributed(use_cuda)
    rank = utils.get_rank()
    world_size = utils.get_world_size()

    utils.setup_logging()
    logging.info(f'Run arguments: {args}')

    pad_vocab = utils.pad_vocabulary(args.math)
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME),
                          pad_vocab)

    # Pre-process dataset only on master node
    if rank == 0:
        srcs, tgts, src_lengths, tgt_lengths = load_dataset(tokenizer, args)
    else:
        srcs, tgts, src_lengths, tgt_lengths = None, None, None, None
        time.sleep(30)

    # Broadcast preprocessed dataset to other ranks
    if world_size > 1:
        srcs, tgts, src_lengths, tgt_lengths = broadcast_dataset(
            world_size, rank, args.max_length_train,
            srcs, tgts, src_lengths, tgt_lengths)

    preproc_train_data = PreprocessedDataset(
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        vocab_size=tokenizer.vocab_size,
        )
    os.makedirs(args.preproc_data_dir, exist_ok=True)
    preproc_train_data.write_data(
        os.path.join(args.preproc_data_dir, 'training.bin'),
        (srcs, src_lengths),
        (tgts, tgt_lengths),
        )


if __name__ == "__main__":
    main()
