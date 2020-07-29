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

"""TensorFlow NMT model implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf

# copybara:strip_begin
from REDACTED import REDACTED
from REDACTED.tensorflow.contrib import training as contrib_training
# copybara:strip_end
from REDACTED.tensorflow.contrib.training.python.training import evaluation
from REDACTED.tensorflow.python.ops import control_flow_util
from REDACTED.mlp_log import mlp_log
from REDACTED.nmt import estimator
from REDACTED.nmt.utils import iterator_utils
from REDACTED.nmt.utils import misc_utils as utils
from REDACTED.nmt.utils import vocab_utils

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument(
      "--num_units", type=int, default=1024, help="Network size.")
  parser.add_argument(
      "--num_layers", type=int, default=4, help="Network depth.")
  parser.add_argument("--num_encoder_layers", type=int, default=None,
                      help="Encoder depth, equal to num_layers if None.")
  parser.add_argument("--num_decoder_layers", type=int, default=None,
                      help="Decoder depth, equal to num_layers if None.")
  parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                      help="Number of partitions for embedding vars.")

  # optimizer
  parser.add_argument(
      "--optimizer", type=str, default="adam", help="sgd | adam")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument(
      "--warmup_steps",
      type=int,
      default=200,
      help="How many steps we inverse-decay learning.")
  parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
  parser.add_argument(
      "--decay_start", type=int, default=3000, help="step to start decay")
  parser.add_argument(
      "--decay_interval",
      type=int,
      default=400,
      help="interval steps between 2 decays")
  parser.add_argument(
      "--decay_steps", type=int, default=5, help="number of decays")
  parser.add_argument(
      "--decay_factor", type=float, default=0.66, help="decay rate")

  parser.add_argument(
      "--max_train_epochs", type=int, default=8,
      help="Maximum number of training epochs.")
  parser.add_argument("--num_examples_per_epoch", type=int, default=3442299,
                      help="Number of examples in one epoch")
  parser.add_argument("--label_smoothing", type=float, default=0.1,
                      help=("If nonzero, smooth the labels towards "
                            "1/num_classes."))

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # data
  parser.add_argument(
      "--src", type=str, default="en", help="Source suffix, e.g., en.")
  parser.add_argument(
      "--tgt", type=str, default="de", help="Target suffix, e.g., de.")
  parser.add_argument(
      "--data_dir", type=str, default="", help="Training/eval data directory.")

  parser.add_argument(
      "--train_prefix",
      type=str,
      default="train.tok.clean.bpe.32000",
      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--test_prefix",
      type=str,
      default="newstest2014",
      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--use_preprocessed_data",
      type="bool",
      default=True,
      help="Whether to use preprocessed training data.")

  parser.add_argument(
      "--out_dir", type=str, default=None, help="Store log/model files.")

  # Vocab
  parser.add_argument(
      "--vocab_prefix",
      type=str,
      default="vocab.bpe.32000",
      help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)

  parser.add_argument("--check_special_token", type="bool", default=True,
                      help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

  # Sequence lengths
  parser.add_argument(
      "--src_max_len",
      type=int,
      default=48,
      help="Max length of src sequences during training.")
  parser.add_argument(
      "--tgt_max_len",
      type=int,
      default=48,
      help="Max length of tgt sequences during training.")
  parser.add_argument(
      "--src_max_len_infer",
      type=int,
      default=160,
      help="Max length of src sequences during inference.")
  parser.add_argument(
      "--tgt_max_len_infer",
      type=int,
      default=160,
      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change)
  parser.add_argument("--forget_bias", type=float, default=0.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=5,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument(
      "--num_buckets",
      type=int,
      default=5,
      help="Put data into similar-length buckets.")
  parser.add_argument(
      "--choose_buckets",
      type=int,
      default=1,
      help="Choose from this number of length buckets per training step.")

  # SPM
  parser.add_argument("--subword_option", type=str, default="bpe",
                      choices=["", "bpe", "spm"],
                      help="""\
                      Set to bpe or spm to activate subword desegmentation.\
                      """)

  # Misc
  parser.add_argument(
      "--num_shards", type=int,
      default=8, help="Number of shards (TPU cores).")
  parser.add_argument(
      "--num_shards_per_host", type=int,
      default=8, help="Number of shards (TPU cores) per host.")
  parser.add_argument(
      "--num_gpus", type=int, default=4, help="Number of gpus in each worker.")
  parser.add_argument(
      "--num_infeed_workers",
      type=int,
      default=1,
      help="Number of TPU workers used for input generation.")
  parser.add_argument(
      "--num_tpu_workers",
      type=int,
      default=1,
      help="Number of TPU workers; if set, uses the distributed-sync pipeline.")
  parser.add_argument("--hparams_path", type=str, default=None,
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument(
      "--random_seed",
      type=int,
      default=None,
      help="Random seed (>0, set a specific seed).")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument(
      "--infer_batch_size",
      type=int,
      default=512,
      help="Batch size for inference mode.")
  parser.add_argument(
      "--examples_to_infer",
      type=int,
      default=3003,
      help="Number of examples to infer.")
  parser.add_argument("--detokenizer_file", type=str,
                      default="mosesdecoder/scripts/tokenizer/detokenizer.perl",
                      help=("""Detokenizer script file."""))
  parser.add_argument("--use_REDACTED", type=bool, default=False)
  parser.add_argument(
      "--target_bleu", type=float, default=24.0, help="Target accuracy.")

  # Advanced inference arguments
  parser.add_argument("--infer_mode", type=str, default="beam_search",
                      choices=["greedy", "sample", "beam_search"],
                      help="Which type of decoder to use during inference.")
  parser.add_argument("--beam_width", type=int, default=5,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument(
      "--length_penalty_weight",
      type=float,
      default=0.6,
      help="Length penalty for beam search.")
  parser.add_argument(
      "--coverage_penalty_weight",
      type=float,
      default=0.1,
      help="Coverage penalty for beam search.")

  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")

  # TPU
  parser.add_argument("--use_tpu", type=bool, default=True)
  parser.add_argument("--master", type=str, default="",
                      help=("Address of the master. Either --master or "
                            "--tpu_name must be specified."))
  parser.add_argument("--tpu_name", type=str, default=None,
                      help=("Name of the TPU for Cluster Resolvers. Either "
                            "--tpu_name or --master must be specified."))
  parser.add_argument("--use_dynamic_rnn", type=bool, default=False)
  parser.add_argument("--use_synthetic_data", type=bool, default=False)
  parser.add_argument(
      "--mode",
      type=str,
      default="train_and_eval",
      choices=["train", "train_and_eval", "infer", "preprocess"])
  parser.add_argument(
      "--activation_dtype",
      type=str,
      default="bfloat16",
      choices=["float32", "bfloat16"])
  parser.add_argument("--tpu_job_name", type=str, default=None)

  # copybara:strip_begin
  # Vizier
  parser.add_argument("--client_handle", type=str, default="",
                      help=("Client_handle for the tuner."))
  parser.add_argument("--study_name", type=str, default=None,
                      help=("Name of Vizier hparams tuning study."))
  parser.add_argument("--REDACTED", type=int,
                      default=REDACTED.StudyConfig.RANDOM_SEARCH,
                      help=("Vizier search algorithm to use."))
  # copybara:strip_end


def create_hparams(flags):
  """Create training hparams."""
  return contrib_training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=flags.data_dir + flags.train_prefix,
      test_prefix=flags.data_dir + flags.test_prefix,
      vocab_prefix=flags.data_dir + flags.vocab_prefix,
      out_dir=flags.out_dir,

      # Networks
      num_units=flags.num_units,
      num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
      num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
      dropout=flags.dropout,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Train
      optimizer=flags.optimizer,
      max_train_epochs=flags.max_train_epochs,
      num_examples_per_epoch=flags.num_examples_per_epoch,
      batch_size=flags.batch_size,
      num_train_steps=int(flags.num_examples_per_epoch / flags.batch_size *
                          flags.max_train_epochs),
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      label_smoothing=flags.label_smoothing,
      warmup_steps=flags.warmup_steps,
      warmup_scheme=flags.warmup_scheme,
      decay_start=flags.decay_start,
      decay_interval=flags.decay_interval,
      decay_steps=flags.decay_steps,
      decay_factor=flags.decay_factor,

      # Data constraints
      num_buckets=flags.num_buckets,
      choose_buckets=flags.choose_buckets,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,
      use_preprocessed_data=flags.use_preprocessed_data,

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size,
      examples_to_infer=flags.examples_to_infer,
      detokenizer_file=flags.data_dir + flags.detokenizer_file,
      use_REDACTED=flags.use_REDACTED,
      target_bleu=flags.target_bleu,

      # Advanced inference arguments
      infer_mode=flags.infer_mode,
      beam_width=flags.beam_width,
      length_penalty_weight=flags.length_penalty_weight,
      coverage_penalty_weight=flags.coverage_penalty_weight,

      # Vocab
      sos=vocab_utils.SOS,
      eos=vocab_utils.EOS,
      subword_option=flags.subword_option,
      check_special_token=flags.check_special_token,

      # Misc
      forget_bias=flags.forget_bias,
      num_shards=flags.num_shards,
      num_shards_per_host=flags.num_shards_per_host,
      num_gpus=flags.num_gpus,
      num_infeed_workers=flags.num_infeed_workers,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      random_seed=flags.random_seed,

      # TPU
      use_tpu=flags.use_tpu,
      master=flags.master,
      tpu_name=flags.tpu_name,
      use_dynamic_rnn=flags.use_dynamic_rnn,
      use_synthetic_data=flags.use_synthetic_data,
      mode=flags.mode,
      activation_dtype=flags.activation_dtype,
      tpu_job_name=flags.tpu_job_name)


def _add_argument(hparams, key, value, update=True):
  """Add an argument to hparams; if exists, change the value if update==True."""
  if hasattr(hparams, key):
    if update:
      setattr(hparams, key, value)
  else:
    hparams.add_hparam(key, value)


def extend_hparams(hparams):
  """Add new arguments to hparams."""
  # Sanity checks
  if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
    raise ValueError("subword option must be either spm, or bpe")
  if hparams.infer_mode == "beam_search" and hparams.beam_width <= 0:
    raise ValueError("beam_width must greater than 0 when using beam_search"
                     "decoder.")

  # Different number of encoder / decoder layers
  assert hparams.num_encoder_layers == hparams.num_decoder_layers

  # The first unidirectional layer (after the bi-directional layer) in
  # the GNMT encoder can't have residual connection due to the input is
  # the concatenation of fw_cell and bw_cell's outputs.
  num_encoder_residual_layers = hparams.num_encoder_layers - 2
  num_decoder_residual_layers = num_encoder_residual_layers
  _add_argument(hparams, "num_encoder_residual_layers",
                num_encoder_residual_layers)
  _add_argument(hparams, "num_decoder_residual_layers",
                num_decoder_residual_layers)

  ## Vocab
  # Get vocab file names first
  if hparams.vocab_prefix:
    src_vocab_file = six.ensure_str(
        hparams.vocab_prefix) + "." + six.ensure_str(hparams.src)
    tgt_vocab_file = six.ensure_str(
        hparams.vocab_prefix) + "." + six.ensure_str(hparams.tgt)
  else:
    raise ValueError("hparams.vocab_prefix must be provided.")

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.out_dir,
      check_special_token=hparams.check_special_token,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  utils.print_out("  using source vocab for target")
  tgt_vocab_file = src_vocab_file
  tgt_vocab_size = src_vocab_size
  _add_argument(hparams, "src_vocab_size", src_vocab_size)
  _add_argument(hparams, "tgt_vocab_size", tgt_vocab_size)
  _add_argument(hparams, "src_vocab_file", src_vocab_file)
  _add_argument(hparams, "tgt_vocab_file", tgt_vocab_file)

  # Num embedding partitions
  _add_argument(
      hparams, "num_enc_emb_partitions", hparams.num_embeddings_partitions)
  _add_argument(
      hparams, "num_dec_emb_partitions", hparams.num_embeddings_partitions)

  # Pretrained Embeddings
  _add_argument(hparams, "src_embed_file", "")
  _add_argument(hparams, "tgt_embed_file", "")

  return hparams


def create_or_load_hparams(default_hparams, hparams_path):
  """Create hparams or load hparams from out_dir."""
  hparams = utils.maybe_parse_standard_hparams(default_hparams, hparams_path)
  hparams = extend_hparams(hparams)
  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def prepare_dataset(flags):
  """Generate the preprocessed dataset."""
  src_file = "%s.%s" % (flags.data_dir + flags.train_prefix, flags.src)
  tgt_file = "%s.%s" % (flags.data_dir + flags.train_prefix, flags.tgt)
  vocab_file = flags.data_dir + flags.vocab_prefix
  _, vocab_file = vocab_utils.check_vocab(vocab_file, flags.out_dir)
  out_file = six.ensure_str(flags.out_dir) + "preprocessed_dataset"
  src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(vocab_file)
  src_dataset = tf.data.TextLineDataset(src_file)
  tgt_dataset = tf.data.TextLineDataset(tgt_file)
  iterator = iterator_utils.get_iterator(
      src_dataset,
      tgt_dataset,
      src_vocab_table,
      tgt_vocab_table,
      batch_size=1,
      global_batch_size=1,
      sos=vocab_utils.SOS,
      eos=vocab_utils.EOS,
      random_seed=1,
      num_buckets=flags.num_buckets,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,
      filter_oversized_sequences=True,
      return_raw=True).make_initializable_iterator()

  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    try:
      i = 0
      while True:
        with open(out_file + "_%d" % i, "wb") as f:
          i += 1
          for _ in range(100):
            for j in sess.run(iterator.get_next()):
              tf.logging.info(j)
              f.write(bytearray(j))
    except tf.errors.OutOfRangeError:
      pass


def run_main(flags, default_hparams, estimator_fn):
  """Run main."""
  # Job
  jobid = flags.jobid
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)
    tf.set_random_seed(random_seed)

  mlp_log.mlperf_print("cache_clear", True)
  mlp_log.mlperf_print("init_start", None)
  mlp_log.mlperf_print("submission_benchmark", "resnet")
  mlp_log.mlperf_print("submission_division", "closed")
  mlp_log.mlperf_print("submission_org", "google")
  mlp_log.mlperf_print("submission_platform", "tpu-v3-%d" % FLAGS.num_shards)
  mlp_log.mlperf_print("submission_status", "research")

  mlp_log.mlperf_print("global_batch_size", FLAGS.batch_size)
  mlp_log.mlperf_print("opt_learning_rate_alt_decay_func", "True")
  mlp_log.mlperf_print("opt_base_learning_rate", FLAGS.learning_rate)
  mlp_log.mlperf_print("opt_learning_rate_decay_interval", FLAGS.decay_interval)
  mlp_log.mlperf_print("opt_learning_rate_decay_factor", FLAGS.decay_factor)
  mlp_log.mlperf_print("opt_learning_rate_decay_steps", FLAGS.decay_steps)
  mlp_log.mlperf_print("opt_learning_rate_remain_steps", FLAGS.decay_start)
  mlp_log.mlperf_print("opt_learning_rate_alt_warmup_func", FLAGS.warmup_scheme)
  mlp_log.mlperf_print("opt_learning_rate_warmup_steps", FLAGS.warmup_steps)
  mlp_log.mlperf_print(
      "max_sequence_length", FLAGS.src_max_len, metadata={"method": "discard"})
  mlp_log.mlperf_print("train_samples", FLAGS.num_examples_per_epoch)
  mlp_log.mlperf_print("eval_samples", FLAGS.examples_to_infer)

  # Model output directory
  out_dir = flags.out_dir
  if out_dir and not tf.gfile.Exists(out_dir):
    utils.print_out("# Creating output directory %s ..." % out_dir)
    tf.gfile.MakeDirs(out_dir)

  # Load hparams.
  hparams = create_or_load_hparams(default_hparams, flags.hparams_path)

  # Train or Evaluation
  return estimator_fn(hparams)


def main(unused_argv):
  # pylint: disable=g-long-lambda
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

  if FLAGS.mode == "preprocess":
    prepare_dataset(FLAGS)
  elif FLAGS.mode == "train":
    print("Running training mode.")
    default_hparams = create_hparams(FLAGS)
    run_main(FLAGS, default_hparams, estimator.train_fn)
  elif FLAGS.mode == "train_and_eval":
    print("Running training and evaluation mode.")
    default_hparams = create_hparams(FLAGS)
    run_main(FLAGS, default_hparams,
             estimator.train_and_eval_with_low_level_api)
  else:
    print("Running inference mode.")
    default_hparams = create_hparams(FLAGS)
    current_epoch = 0
    last_step = 0
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(FLAGS.out_dir):
      # Terminate eval job once target score is reached
      current_step = int(six.ensure_str(os.path.basename(ckpt)).split("-")[1])
      if current_step <= last_step:
        continue
      last_step = current_step
      tf.logging.info("Starting to evaluate...%s", ckpt)
      try:
        score = run_main(FLAGS, default_hparams, estimator.eval_fn)
        current_epoch += 1
        if score > FLAGS.target_bleu:
          tf.logging.info(
              "Evaluation finished after training step %d" % current_step)
          break
        # Terminate eval job when final checkpoint is reached
        max_steps = default_hparams.num_train_steps
        if current_step >= max_steps:
          tf.logging.info(
              "Evaluation finished but failed to reach target score.")
          break

      except tf.errors.NotFoundError:
        tf.logging.info(
            "Checkpoint %s no longer exists, skipping checkpoint" % ckpt)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
