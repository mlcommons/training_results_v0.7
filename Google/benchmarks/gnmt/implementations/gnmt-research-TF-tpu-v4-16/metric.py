# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Estimator functions supporting running on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import subprocess
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.python.ops import lookup_ops
from REDACTED.tensorflow_models.mlperf.models.rough.nmt.utils import evaluation_utils
from REDACTED.tensorflow_models.mlperf.models.rough.nmt.utils import misc_utils
from REDACTED.tensorflow_models.mlperf.models.rough.nmt.utils import nmt_utils
from REDACTED.tensorflow_models.mlperf.models.rough.nmt.utils import vocab_utils


def get_sacrebleu(trans_file, detokenizer_file):
  """Detokenize the trans_file and get the sacrebleu score."""
  assert tf.gfile.Exists(detokenizer_file)
  local_detokenizer_file = "/tmp/detokenizer.perl"
  if not tf.gfile.Exists(local_detokenizer_file):
    tf.gfile.Copy(detokenizer_file, local_detokenizer_file)

  assert tf.gfile.Exists(trans_file)
  local_trans_file = "/tmp/newstest2014_out.tok.de"
  if tf.gfile.Exists(local_trans_file):
    tf.gfile.Remove(local_trans_file)
  tf.gfile.Copy(trans_file, local_trans_file)

  detok_trans_path = "/tmp/newstest2014_out.detok.de"
  if tf.gfile.Exists(detok_trans_path):
    tf.gfile.Remove(detok_trans_path)

  # Detokenize the trans_file.
  cmd = "cat %s | perl %s -l de | cat > %s" % (
      local_trans_file, local_detokenizer_file, detok_trans_path)
  subprocess.run(cmd, shell=True)
  assert tf.gfile.Exists(detok_trans_path)

  # run sacrebleu
  cmd = ("cat %s | sacrebleu -t wmt14/full -l en-de --score-only -lc --tokenize"
         " intl") % (
             detok_trans_path)
  sacrebleu = subprocess.run([cmd], stdout=subprocess.PIPE, shell=True)
  return float(sacrebleu.stdout.strip())


def _convert_ids_to_strings(tgt_vocab_file, ids):
  """Convert prediction ids to words."""
  with tf.Session() as sess:
    reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)
    sess.run(tf.tables_initializer())
    translations = sess.run(
        reverse_target_vocab_table.lookup(
            tf.to_int64(tf.convert_to_tensor(np.asarray(ids)))))
  return translations


def get_metric(hparams, predictions, current_step):
  """Run inference and compute metric."""
  predicted_ids = []
  for prediction in predictions:
    predicted_ids.append(prediction["predictions"])

  if hparams.examples_to_infer < len(predicted_ids):
    predicted_ids = predicted_ids[0:hparams.examples_to_infer]
  translations = _convert_ids_to_strings(hparams.tgt_vocab_file, predicted_ids)

  trans_file = os.path.join(hparams.out_dir,
                            "newstest2014_out_{}.tok.de".format(current_step))
  trans_dir = os.path.dirname(trans_file)
  if not tf.gfile.Exists(trans_dir):
    tf.gfile.MakeDirs(trans_dir)
  tf.logging.info("Writing to file %s" % trans_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(trans_file,
                                                mode="wb")) as trans_f:
    trans_f.write("")  # Write empty string to ensure file is created.
    for translation in translations:
      sentence = nmt_utils.get_translation(
          translation,
          tgt_eos=hparams.eos,
          subword_option=hparams.subword_option)
      trans_f.write((sentence + b"\n").decode("utf-8"))

  # Evaluation
  output_dir = os.path.join(hparams.out_dir, "eval")
  tf.gfile.MakeDirs(output_dir)
  summary_writer = tf.summary.FileWriter(output_dir)

  ref_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)

  if hparams.use_REDACTED:
    score = evaluation_utils.evaluate(ref_file, trans_file)
  else:
    score = get_sacrebleu(trans_file, hparams.detokenizer_file)
  with tf.Graph().as_default():
    summaries = []
    summaries.append(tf.Summary.Value(tag="sacrebleu", simple_value=score))
  tf_summary = tf.Summary(value=list(summaries))
  summary_writer.add_summary(tf_summary, current_step)

  misc_utils.print_out("  %s: %.1f" % ("sacrebleu", score))

  summary_writer.close()
  return score
