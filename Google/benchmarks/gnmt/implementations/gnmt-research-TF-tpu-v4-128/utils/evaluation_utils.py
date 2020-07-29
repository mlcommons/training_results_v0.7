# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""

from __future__ import print_function

import codecs
import REDACTED
from . import moses_decoder
from . import sacrebleu

import tensorflow.compat.v1 as tf

__all__ = ["evaluate"]


def evaluate(ref_file, trans_file, lower_case=True):
  """Computes sacreBLEU score."""
  ref_stream = []
  trans_stream = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
    ref_stream.extend(fh.readlines())
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    trans_stream.extend(moses_decoder.detokenize(fh.readlines()))
  assert len(ref_stream) == len(trans_stream)
  return sacrebleu.corpus_bleu(
      trans_stream, [ref_stream],
      smooth_method="floor",
      smooth_value=sacrebleu.SMOOTH_VALUE_DEFAULT,
      force=True,
      lowercase=lower_case,
      tokenize="intl",
      use_effective_order=True).score
