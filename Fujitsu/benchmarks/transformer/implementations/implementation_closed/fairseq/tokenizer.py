# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import re

import torch


SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


# This is just to binarize UTF-8 files containing the already-tokenized integer IDs (Reading raw data is sloooowwwwww.)
class MockTokenizer:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, mode='r', newline='\n') as f:
            for line in f:
                for tok in line.split(' '):
                    dict.add_symbol(tok)
                #dict.add_symbol(dict.eos_word)  # Already added by MLPerf preprocessing

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            pass

        with open(filename, mode='r', newline='\n') as f:
            for line in f:
                nwords = len(line.split(' '))
                ids = torch.IntTensor(nwords).long()
                for idx, tok in enumerate(line.split(' ')):
                    ids[idx] = int(tok)

                nseq += 1

                consumer(ids)
                ntok += len(ids)

        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False, lowercase=False):
        return 0    # Do not call this....

class Tokenizer:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            pass

        with open(filename, 'r') as f:
            for line in f:
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False, lowercase=False):
        words = tokenize(line)
        if lowercase:
            lc_words = []
            for word in words:
                lc_words.append(word.lower())
            words = lc_words
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids
