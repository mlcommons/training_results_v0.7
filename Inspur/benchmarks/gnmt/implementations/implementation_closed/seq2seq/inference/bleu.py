from itertools import zip_longest

import sacrebleu
import torch


def read_reference(fname, indices):
    with open(fname) as f:
        refs = f.readlines()
    refs = [refs[i] for i in indices]
    return refs


def all_reduce(val):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        val = torch.tensor(val)

        if hasattr(torch.distributed, "get_backend"):
            _backend = torch.distributed.get_backend()
            if hasattr(torch.distributed, "DistBackend"):
                backend_enum_holder = torch.distributed.DistBackend
            else:
                backend_enum_holder = torch.distributed.Backend
        else:
            _backend = torch.distributed._backend
            backend_enum_holder = torch.distributed.dist_backend

        if _backend == backend_enum_holder.NCCL:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        val = val.to(device)
        torch.distributed.all_reduce(val)
        val = val.tolist()
    return val


def corpus_bleu(sys_stream, ref_streams, smooth='exp', smooth_floor=0.0,
                force=False, lowercase=False,
                tokenize=sacrebleu.DEFAULT_TOKENIZER,
                use_effective_order=False) -> sacrebleu.BLEU:
    """Produces BLEU scores along with its sufficient statistics from a source
    against one or more references.

    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a
                        sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_floor: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """

    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(sacrebleu.NGRAM_ORDER)]
    total = [0 for n in range(sacrebleu.NGRAM_ORDER)]

    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different "
                           "lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        output, *refs = [sacrebleu.TOKENIZERS[tokenize](x.rstrip()) for x in
                         lines]

        ref_ngrams, closest_diff, closest_len = sacrebleu.ref_stats(output,
                                                                    refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = sacrebleu.extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n-1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n-1] += sys_ngrams[ngram]

    correct = all_reduce(correct)
    total = all_reduce(total)
    sys_len = all_reduce(sys_len)
    ref_len = all_reduce(ref_len)

    return sacrebleu.compute_bleu(correct, total, sys_len, ref_len, smooth,
                                  smooth_floor, use_effective_order)


def compute_bleu(output, indices, ref_fname):
    refs = read_reference(ref_fname, indices)
    bleu = corpus_bleu(output, [refs], lowercase=True,
                       tokenize='intl')
    return bleu
