# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import struct

import numpy as np
import torch

from fairseq.tokenizer import Tokenizer


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self, path):
        super().__init__()
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == b'TNTIDX\x00\x00'
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self.size, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)
        self.read_data(path)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __del__(self):
        self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        #a += 1  ## DEBUG: lua_index_compat
        item = torch.from_numpy(a).long()
        return item

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and
            os.path.exists(data_file_path(path))
        )


class MockedInMemoryDataset(IndexedDataset):
    """Loader for TorchNet IndexedDataset, keeps all the data in memory"""

    def __init__(self, path, n_seq_pairs_in_mock_data, uniform_n_seq_per_batch, uniform_seq_len_per_batch):
        self.dtype = np.int64
        self.uniform_n_seq_per_batch = uniform_n_seq_per_batch
        self.uniform_seq_len_per_batch = uniform_seq_len_per_batch
        self.size = n_seq_pairs_in_mock_data

        self.sizes = []
        for i in range(n_seq_pairs_in_mock_data):
            self.sizes.append(uniform_seq_len_per_batch)

    def __del__(self):
        pass

    def __getitem__(self, i):
        self.check_index(i)
        arbitrary_token_id = 55    # Just not a reserved token
        a = np.ones((self.uniform_seq_len_per_batch,), dtype=self.dtype) * arbitrary_token_id
        a[-1] = 2  # Manually add an <EOS>
        #a[self.uniform_seq_len_per_batch-1] = 2    # Manually add an <EOS>
        return torch.from_numpy(a).long()


class IndexedInMemoryDataset(IndexedDataset):
    """Loader for TorchNet IndexedDataset, keeps all the data in memory"""

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb')
        self.buffer = np.empty(self.data_offsets[-1], dtype=self.dtype)
        self.data_file.readinto(self.buffer)
        #print('buffer max:', np.max(self.buffer), np.min(self.buffer))
        #self.buffer[self.buffer > 0] += 1  ## DEBUG
        #self.buffer += 1 ## DEBUG
        #print('buffer max after:', np.max(self.buffer), np.min(self.buffer))
        self.data_file.close()

    def __del__(self):
        pass

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        np.copyto(a, self.buffer[self.data_offsets[i]:self.data_offsets[i + 1]])
        return torch.from_numpy(a).long()


class IndexedRawTextDataset(IndexedDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = Tokenizer.tokenize(
                    line, dictionary, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedRawTokenIDDataset(IndexedDataset):
    """Takes a text file containing token IDs (integers written in UTF-8 format) as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r') as f:
            for line in f:
                if line != '\n':
                    self.lines.append(line.strip('\n'))
                    nwords = len(line.split(' '))
                    tokens = torch.IntTensor(nwords).long()
                    for idx, tok in enumerate(line.split(' ')):
                        tokens[idx] = int(tok)

                    #tokens = line.split(' ')
                    self.tokens_list.append(tokens)
                    self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()
