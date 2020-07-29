from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import random
import sys
import tarfile
import urllib

import six
import urllib.request

from fairseq.data import indexed_dataset
from fairseq.data import dictionary
from fairseq.tokenizer import MockTokenizer


def make_binary_dataset(data_dir):
    dict = dictionary.Dictionary.load(data_dir + '/dict.en.txt')

    print('Converting utf8 files to fairseq binary')
    files = glob.glob(data_dir + '/utf8/test*.en', recursive=True)
    files += glob.glob(data_dir + '/utf8/test*.de', recursive=True)

    files += glob.glob(data_dir + '/utf8/dev*.en', recursive=True)
    files += glob.glob(data_dir + '/utf8/dev*.de', recursive=True)

    files += glob.glob(data_dir + '/utf8/train*.en', recursive=True)
    files += glob.glob(data_dir + '/utf8/train*.de', recursive=True)

    def consumer(tensor):
        ds.add_item(tensor)

    for file in files:
        print('Converting file:', file)
        ds = indexed_dataset.IndexedDatasetBuilder(file + '.bin')

        def consumer(tensor):
            ds.add_item(tensor)

        res = MockTokenizer.binarize(file, dict, consumer)

        ds.finalize(file + '.idx')


def main(unused_argv):
  make_binary_dataset(FLAGS.data_dir)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/research/transformer/processed_data",
      help="[default: %(default)s] Directory for where the "
           "translate_ende_wmt32k dataset is saved.",
      metavar="<DD>")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
