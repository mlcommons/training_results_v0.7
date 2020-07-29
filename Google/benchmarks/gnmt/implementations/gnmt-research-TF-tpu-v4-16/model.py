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
"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import tensorflow.compat.v1 as tf
from REDACTED.tensorflow.contrib import learn as contrib_learn
from REDACTED.tensorflow.contrib import seq2seq as contrib_seq2seq

from REDACTED.tensorflow.python.data.util import nest
from REDACTED.tensorflow.python.ops import inplace_ops
from REDACTED.tensorflow_models.mlperf.models.rough.nmt import beam_search_decoder
from REDACTED.tensorflow_models.mlperf.models.rough.nmt import decoder
from REDACTED.tensorflow_models.mlperf.models.rough.nmt import model_helper
from REDACTED.tensorflow_models.mlperf.models.rough.nmt.utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel"]


def dropout(shape, dtype, keep_ratio):
  """Dropout helper function."""
  return tf.math.floor(tf.random.uniform(shape, dtype=dtype) +
                       keep_ratio) / keep_ratio


def lstm_cell_split(gate_inputs, c, padding):
  """Helper function to perform inexpensive activation of lstm cell."""
  # i = input_gate, j = new_input, f = forget_gate, o = output_gate
  i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

  new_c = c * tf.math.sigmoid(f) + tf.math.sigmoid(i) * tf.math.tanh(j)
  new_h = tf.math.tanh(new_c) * tf.math.sigmoid(o)
  if padding is not None:
    new_c = new_c * tf.expand_dims(padding, 1)
    new_h = new_h * tf.expand_dims(padding, 1)
  new_state = {"c": new_c, "h": new_h}
  return new_state


def lstm_cell_gate(theta, h, inputs):
  """Helper function to performan expensive matmul of lstm cell."""
  kernel, bias = theta["kernel"], theta["bias"]
  gate_inputs = tf.matmul(h, kernel) + inputs
  gate_inputs = tf.nn.bias_add(gate_inputs, bias)
  return gate_inputs


def lstm_cell_grad(theta, state0, inputs, extras, dstate1):
  """Gradient function for lstm_cell."""
  padding = inputs["padding"] if (inputs is not None and
                                  "padding" in inputs) else None
  state1 = nest.flatten(lstm_cell_split(extras, state0["c"], padding))
  dstate1 = nest.flatten(dstate1)
  grad = tf.gradients(state1, [extras], dstate1)[0]
  dtheta = {
      "bias": tf.reduce_sum(grad, 0)
  }
  dinputs = {"rnn": grad}
  dstate = {"c": tf.gradients(state1, state0["c"], dstate1)[0]}
  dstate["h"] = tf.matmul(grad, tf.transpose(theta["kernel"]))
  if padding is not None:
    dinputs["padding"] = padding
  return dtheta, dstate, dinputs


def lstm_cell(theta, state, inputs):
  c, h = state["c"], state["h"]
  gate_inputs = lstm_cell_gate(theta, h, inputs["rnn"])
  padding = inputs["padding"] if "padding" in inputs else None
  return lstm_cell_split(gate_inputs, c, padding), gate_inputs


def attention(theta, new_lstm_state):
  """Helper function to add attention."""
  lstm_output = new_lstm_state["h"]
  query = tf.expand_dims(tf.matmul(lstm_output, theta["query_kernel"]), 0)
  normed_v = theta["atten_g"] * theta["atten_v"] * tf.rsqrt(
      tf.reduce_sum(tf.square(theta["atten_v"])))
  score = tf.reduce_sum(
      normed_v * tf.tanh(theta["keys"] + query + theta["atten_b"]), [2])
  score = tf.transpose(score)
  score = tf.where(
      theta["seq_mask"] > 0.5, score,
      tf.ones_like(score) * tf.as_dtype(score.dtype).as_numpy_dtype(-np.inf))

  alignments = tf.nn.softmax(score)
  score = tf.transpose(alignments)

  atten = tf.reduce_sum(tf.expand_dims(score, 2) * theta["values"], 0)
  new_states = {
      "attention": atten,
      "alignments": alignments
  }
  for k in new_lstm_state:
    new_states[k] = new_lstm_state[k]
  return new_states


def attention_cell_grad(theta, state0, unused_inputs, extras, dstate1):
  """Gradient function for attention_cell."""
  new_lstm_state = lstm_cell_split(extras, state0["c"], None)
  new_states = attention(theta, new_lstm_state)
  del new_states["alignments"]

  y = nest.flatten(new_states)
  x = [extras, state0["c"]] + nest.flatten(theta)
  dy = nest.flatten(dstate1)
  g = tf.gradients(y, x, dy)
  dtheta = nest.pack_sequence_as(theta, g[2:])
  grad, dstate_c = g[:2]

  dtheta["bias"] = tf.reduce_sum(grad, 0)

  datten = tf.matmul(grad, tf.transpose(theta["attention_kernel"]))
  dstate_h = tf.matmul(grad, tf.transpose(theta["kernel"]))

  dstate = {
      "h": dstate_h,
      "c": dstate_c,
      "attention": datten,
  }
  return dtheta, dstate, {"rnn": grad}


def attention_cell(theta, state, inputs):
  """Attention cell followed by LSTM cells."""
  lstm_input = {
      "rnn":
          inputs["rnn"] +
          tf.matmul(state["attention"], theta["attention_kernel"])
  }
  new_lstm_state, gate = lstm_cell(theta, state, lstm_input)
  return attention(theta, new_lstm_state), gate


def build_rnn(orig_theta,
              state0,
              orig_inputs,
              cell_fn,
              cell_grad,
              max_length,
              reverse=False):
  """Helper function to build an RNN."""
  max_time, batch_size = orig_inputs["rnn"].shape.as_list()[:2]
  skipped_theta = ["kernel", "attention_kernel", "memory_kernel", "seq_mask"]
  skipped_state = ["alignments"]

  @tf.custom_gradient
  def _rnn(*inp):
    """Function that drives RNN with early stop."""
    inputs = nest.pack_sequence_as(orig_inputs, inp[0:len(orig_inputs)])
    theta = nest.pack_sequence_as(orig_theta, inp[len(orig_inputs):])

    def _cell_fn(theta, state0, acc_state, acc_gate, i):
      """RNN cell function."""
      input_slice = {k: tf.gather(inputs[k], i) for k in inputs}
      state1, gate = cell_fn(theta, state0, input_slice)
      for k in state0:
        if k not in skipped_state:
          acc_state[k] = tf.stop_gradient(
              inplace_ops.alias_inplace_update(acc_state[k], i, state1[k]))
      acc_gate = tf.stop_gradient(
          inplace_ops.alias_inplace_update(acc_gate, i, gate))
      return theta, state1, acc_state, acc_gate, i - 1 if reverse else i + 1

    def _should_continue(i, is_backward=False):
      if is_backward:
        return i < max_length - 1 if reverse else i > 0
      else:
        return i >= 0 if reverse else i < max_length

    acc_state = {
        k: tf.zeros([max_time, batch_size, state0["c"].shape[-1]],
                    state0["c"].dtype) for k in state0 if k not in skipped_state
    }
    acc_state, acc_gate = tf.while_loop(
        lambda theta, state0, acc_state, acc_gate, i: _should_continue(i),
        _cell_fn, [
            theta, state0, acc_state,
            tf.zeros_like(inputs["rnn"]),
            max_length - 1 if reverse else tf.zeros([], tf.int32)
        ])[2:4]
    ret = {"h": acc_state["h"]}
    if "attention" in acc_state:
      ret["attention"] = acc_state["attention"]

    def _cell_grad_fn_with_state0(state0, theta, dy, dstate1, dtheta, dinput,
                                  i):
      """Gradient cell function."""
      state0 = {
          k: tf.stop_gradient(state0[k])
          for k in state0
          if k not in skipped_state
      }
      theta = {k: tf.stop_gradient(theta[k]) for k in theta}
      if "padding" in inputs:
        inputs_slice = {"padding": tf.gather(inputs["padding"], i)}
      else:
        inputs_slice = None
      gate = tf.gather(acc_gate, i)
      for k in dy:
        dstate1[k] = dstate1[k] + tf.gather(dy[k], i)
      dt, dstate, di = cell_grad(theta, state0, inputs_slice, gate, dstate1)
      dtheta = {k: dtheta[k] + dt[k] for k in dtheta if k not in skipped_theta}
      dinput = {
          k: inplace_ops.alias_inplace_update(dinput[k], i, di[k]) for k in di
      }
      return theta, dy, dstate, dtheta, dinput, i + 1 if reverse else i - 1

    def _cell_grad_fn(theta, dy, dstate1, dtheta, dinput, i):
      """Gradient cell function wrapper."""
      return _cell_grad_fn_with_state0(
          {
              k: tf.gather(acc_state[k], i + 1 if reverse else i - 1)
              for k in acc_state
          }, theta, dy, dstate1, dtheta, dinput, i)

    def grad(*dy):
      """Gradient function for build_rnn."""
      dy = nest.pack_sequence_as(ret, dy)

      def _continue(unused_theta, unused_dy, unused_dstate1, unused_dtheta,
                    unused_dinput, i):
        return _should_continue(i, True)

      dstate1, dtheta, dinput = tf.while_loop(_continue, _cell_grad_fn, [
          theta,
          dy,
          {
              k: tf.zeros_like(state0[k])
              for k in state0
              if k not in skipped_state
          },
          {k: tf.zeros_like(theta[k]) for k in theta if k not in skipped_theta},
          {k: tf.zeros_like(inputs[k]) for k in inputs},
          tf.zeros([], tf.int32) if reverse else max_length - 1,
      ])[2:5]
      dtheta, dinput = _cell_grad_fn_with_state0(
          state0, theta, dy, dstate1, dtheta, dinput,
          max_length - 1 if reverse else tf.zeros([], dtype=tf.int32))[3:5]
      state0_h = tf.reshape(acc_state["h"], [-1, theta["kernel"].shape[0]])
      state0_atten = tf.reshape(acc_state["attention"],
                                [-1, theta["attention_kernel"].shape[0]
                                ]) if "attention_kernel" in theta else None
      grad = tf.reshape(dinput["rnn"], [-1, theta["kernel"].shape[1]])
      if reverse:
        state0_h = tf.split(state0_h, [batch_size, -1])[1]
        grad = tf.split(grad, [-1, batch_size])[0]
      else:
        if state0_atten is not None:
          state0_atten = tf.split(state0_atten, [-1, batch_size])[0]
        state0_h = tf.split(state0_h, [-1, batch_size])[0]
        grad = tf.split(grad, [batch_size, -1])[1]

      if state0_atten is not None:
        dtheta["attention_kernel"] = tf.matmul(tf.transpose(state0_atten), grad)
      dtheta["kernel"] = tf.matmul(tf.transpose(state0_h), grad)

      if "memory_kernel" in orig_theta:
        dtheta["memory_kernel"] = tf.zeros_like(orig_theta["memory_kernel"])
        dtheta["seq_mask"] = tf.zeros_like(orig_theta["seq_mask"])
      return dinput, dtheta

    return ret, grad

  return dict(
      _rnn(*(tuple(nest.flatten(orig_inputs)) +
             tuple(nest.flatten(orig_theta)))))


def build_uni_rnn(inputs,
                  max_seq_len,
                  num_units,
                  name,
                  reverse=False):
  """Build the uni-directional RNN."""
  theta = {}
  _, batch_size, input_feature_dim = inputs["rnn"].shape
  dtype = inputs["rnn"].dtype

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    theta = {
        "kernel": tf.get_variable("kernel", [num_units, num_units * 4]),
        "bias": tf.get_variable("bias", [num_units * 4])
    }
    state0 = {
        "h": tf.zeros([batch_size, num_units], dtype=dtype),
        "c": tf.zeros([batch_size, num_units], dtype=dtype)
    }
    input_kernel = tf.get_variable("input_kernel",
                                   [input_feature_dim, num_units * 4])

  inp = {"rnn": tf.einsum("tbf,fd->tbd", inputs["rnn"], input_kernel)}
  if "padding" in inputs:
    inp["padding"] = inputs["padding"]

  output = build_rnn(theta, state0, inp, lstm_cell, lstm_cell_grad, max_seq_len,
                     reverse)
  return output["h"]


def build_bid_rnn(inputs, seq_len, num_units, name):
  """Build the bi-directional RNN."""
  max_seq_len = tf.reduce_max(seq_len)
  fwd = build_uni_rnn(inputs, max_seq_len, num_units,
                      name + "/fw/cell_fn/basic_lstm_cell", False)
  bwd_inputs = {k: inputs[k] for k in inputs}
  bwd_inputs["padding"] = tf.transpose(
      tf.sequence_mask(seq_len, inputs["rnn"].shape[0], inputs["rnn"].dtype))
  bwd = build_uni_rnn(bwd_inputs, max_seq_len, num_units,
                      name + "/bw/cell_fn/basic_lstm_cell", True)
  return tf.concat([fwd, bwd], -1)


def build_atten_rnn(encoder_outputs, src_seq_len, num_units, beam_width, name):
  """Build the attention decoder RNN."""
  dtype = encoder_outputs.dtype
  max_time, batch_size, input_feature_dim = encoder_outputs.shape
  if beam_width > 1:
    encoder_outputs = tf.reshape(
        tf.tile(encoder_outputs, [1, 1, beam_width]),
        [max_time, batch_size * beam_width, input_feature_dim])
    src_seq_len = tf.reshape(
        tf.tile(tf.reshape(src_seq_len, [-1, 1]), [1, beam_width]), [-1])
    batch_size = batch_size * beam_width

  with tf.variable_scope("memory_layer", reuse=tf.AUTO_REUSE):
    memory_kernel = tf.get_variable("kernel", [num_units, num_units])

  keys = tf.reshape(
      tf.matmul(
          tf.reshape(encoder_outputs, [max_time * batch_size, -1]),
          memory_kernel), [max_time, batch_size, -1])
  seq_mask = tf.sequence_mask(src_seq_len, max_time, dtype)

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    theta = []
    state0 = []
    input_kernels = []

    with tf.variable_scope("cell_0_attention/attention"):
      with tf.variable_scope("cell_fn/basic_lstm_cell"):
        kernel = tf.get_variable("kernel", [num_units * 1, num_units * 4])
        input_kernel = tf.get_variable("input_kernel",
                                       [num_units, num_units * 4])
        attention_kernel = tf.get_variable("attention_kernel",
                                           [num_units, num_units * 4])
        bias = tf.get_variable("bias", [num_units * 4])

      with tf.variable_scope("bahdanau_attention"):
        with tf.variable_scope("query_layer"):
          query_kernel = tf.get_variable("kernel", [num_units, num_units])
        input_kernels.append(input_kernel)
        theta.append({
            "kernel":
                kernel,
            "attention_kernel":
                attention_kernel,
            "bias":
                bias,
            "memory_kernel":
                memory_kernel,
            "query_kernel":
                query_kernel,
            "atten_v":
                tf.get_variable("attention_v", [num_units]),
            "atten_g":
                tf.get_variable(
                    "attention_g", [],
                    initializer=tf.constant_initializer(
                        math.sqrt(1. / num_units))),
            "atten_b":
                tf.get_variable(
                    "attention_b", [num_units],
                    initializer=tf.zeros_initializer()),
            "keys":
                keys,
            "values":
                encoder_outputs,
            "seq_mask":
                seq_mask
        })
      state0.append({
          "c": tf.zeros([batch_size, num_units], dtype=dtype),
          "h": tf.zeros([batch_size, num_units], dtype=dtype),
          "attention": tf.zeros([batch_size, num_units], dtype=dtype),
          "alignments": tf.zeros([batch_size, max_time], dtype=dtype)
      })
    for i in range(1, 4):
      with tf.variable_scope("cell_%d/cell_fn/basic_lstm_cell" % i):
        theta.append({
            "kernel": tf.get_variable("kernel", [num_units, num_units * 4]),
            "bias": tf.get_variable("bias", [num_units * 4])
        })
        input_kernels.append(
            tf.get_variable("input_kernel", [num_units * 2, num_units * 4]))
        state0.append({
            "c": tf.zeros([batch_size, num_units], dtype=dtype),
            "h": tf.zeros([batch_size, num_units], dtype=dtype),
        })
  return theta, input_kernels, state0


@tf.custom_gradient
def softmax_cross_entropy(logits, label):
  """Helper function to compute softmax cross entropy loss."""
  exp_logits = tf.math.exp(logits)
  sum_exp = tf.reduce_sum(exp_logits, -1)
  log_sum_exp = tf.math.log(sum_exp)
  reduce_logits = tf.reduce_sum(logits * label, -1)
  loss = log_sum_exp - reduce_logits

  def grad(dy):
    return (exp_logits / tf.expand_dims(sum_exp, -1) - label) * tf.expand_dims(
        dy, -1), tf.zeros_like(label)

  return loss, grad


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self, hparams, mode, features):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      features: a dict of input features.
    """
    # Set params
    self._set_params_initializer(hparams, mode, features)
    if self.mode == contrib_learn.ModeKeys.INFER:
      self.build_train_graph(hparams, 0)
    else:
      src_len = tf.reduce_max(self.features["source_sequence_length"])
      tgt_len = tf.reduce_max(self.features["target_sequence_length"])
      max_len = tf.maximum(src_len, tgt_len)
      max_len = tf.maximum(max_len, 12)
      gradients, global_norm = tf.switch_case(
          tf.cast((max_len - 12) / 6, tf.int32), {
              0: lambda: self.build_train_graph(hparams, 18),
              1: lambda: self.build_train_graph(hparams, 24),
              2: lambda: self.build_train_graph(hparams, 30),
              3: lambda: self.build_train_graph(hparams, 36),
              4: lambda: self.build_train_graph(hparams, 42),
          },
          default=lambda: self.build_train_graph(hparams, 0))
      self.learning_rate = tf.constant(hparams.learning_rate)
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      self.learning_rate = self._get_learning_rate_decay(hparams)
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

      gradients, _ = tf.clip_by_global_norm(gradients,
                                            hparams.max_gradient_norm,
                                            global_norm)

      gradients = [(tf.cast(
          tf.tpu.cross_replica_sum(tf.cast(g, tf.bfloat16)), tf.float32), v)
                   for g, v in zip(gradients, tf.trainable_variables())]

      self.update = opt.apply_gradients(gradients, global_step=self.global_step)

  def build_train_graph(self, hparams, length):
    # Train graph
    self.length = length
    with tf.variable_scope("nmt", reuse=tf.AUTO_REUSE):
      self.init_embeddings(hparams)
      res = self.build_graph(hparams)
      self._set_train_or_infer(res)
      if self.mode != contrib_learn.ModeKeys.INFER:
        return self.gradients, tf.global_norm(self.gradients)


  def _emb_lookup(self, weight, index, is_decoder=False):
    return tf.cast(
        tf.reshape(
            tf.gather(weight, tf.reshape(index, [-1])),
            [index.shape[0], index.shape[1], -1]), self.dtype)

  def _set_params_initializer(self, hparams, mode, features):
    """Set various params for self and initialize."""
    self.mode = mode
    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.features = features

    self.dtype = tf.as_dtype(hparams.activation_dtype)

    self.single_cell_fn = None

    # Set num units
    self.num_units = hparams.num_units
    self.eos_id = hparams.tgt_eos_id
    self.label_smoothing = hparams.label_smoothing

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    # Batch size
    self.batch_size = tf.size(self.features["source_sequence_length"])

    # Global step
    # Use get_global_step instead of user-defied global steps. Otherwise the
    # num_train_steps in TPUEstimator.train has no effect (will train forever).
    # TPUestimator only check if tf.train.get_global_step() < num_train_steps
    self.global_step = tf.train.get_or_create_global_step()

    # Initializer
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
        hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.encoder_emb_lookup_fn = (
        self._emb_lookup if self.mode == contrib_learn.ModeKeys.TRAIN else
        tf.nn.embedding_lookup)

  def _set_train_or_infer(self, res):
    """Set up training."""
    if self.mode == contrib_learn.ModeKeys.INFER:
      self.predicted_ids = res[1]

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    if self.mode == contrib_learn.ModeKeys.TRAIN:
      loss = res[0]
      # Gradients
      self.gradients = [
          tf.convert_to_tensor(g) for g in tf.gradients(loss, params)
      ]

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    return tf.cond(
        self.global_step < hparams.decay_start,
        lambda: self.learning_rate,
        lambda: tf.maximum(  # pylint: disable=g-long-lambda
            tf.train.exponential_decay(
                self.learning_rate,
                self.global_step - hparams.decay_start,
                hparams.decay_interval,
                hparams.decay_factor,
                staircase=True),
            self.learning_rate * tf.pow(hparams.decay_factor, hparams.
                                        decay_steps)),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=self.num_units,
            tgt_embed_size=self.num_units,
            num_enc_partitions=hparams.num_enc_emb_partitions,
            num_dec_partitions=hparams.num_dec_emb_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
        ))

  def _build_model(self, hparams):
    """Builds a sequence-to-sequence model.

    Args:
      hparams: Hyperparameter configurations.

    Returns:
      For infrence, A tuple of the form
      (logits, decoder_cell_outputs, predicted_ids),
      where:
        logits: logits output of the decoder.
        decoder_cell_outputs: the output of decoder.
        predicted_ids: predicted ids from beam search.
      For training, returns the final loss
    """
    # Encoder
    self.encoder_outputs = self._build_encoder(hparams)

    ## Decoder
    return self._build_decoder(self.encoder_outputs, hparams)

  def build_graph(self, hparams):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.

    Returns:
      A tuple of the form (logits, predicted_ids) for infererence and
      (loss, None) for training.
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols]
        loss: float32 scalar
        predicted_ids: predicted ids from beam search.
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    # Projection
    with tf.variable_scope("build_network"):
      with tf.variable_scope("decoder/output_projection", reuse=tf.AUTO_REUSE):
        self.output_layer = tf.get_variable(
            "output_projection", [self.num_units, self.tgt_vocab_size])

    with tf.variable_scope(
        "dynamic_seq2seq", dtype=self.dtype, reuse=tf.AUTO_REUSE):
      if hparams.activation_dtype == "bfloat16":
        tf.get_variable_scope().set_custom_getter(
            utils.bfloat16_var_getter if hparams.activation_dtype == "bfloat16"
            else None)
        logits_or_loss, decoder_cell_outputs, predicted_ids = self._build_model(
            hparams)
        if decoder_cell_outputs is not None:
          decoder_cell_outputs = tf.cast(decoder_cell_outputs, tf.float32)
      else:
        logits_or_loss, decoder_cell_outputs, predicted_ids = self._build_model(
            hparams)

    return logits_or_loss, predicted_ids

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(
          tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, encoder_outputs, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      hparams: The Hyperparameters configurations.

    Returns:
      For inference, A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size]
      For training, returns the final loss
    """

    ## Decoder.
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:
      # Optional ops depends on which mode we are in and which loss function we
      # are using.
      logits = tf.no_op()
      decoder_cell_outputs = None
      if self.mode == contrib_learn.ModeKeys.TRAIN:
        beam_width = 1
      else:
        beam_width = hparams.beam_width
      theta, input_kernels, state0 = build_atten_rnn(
          encoder_outputs, self.features["source_sequence_length"],
          hparams.num_units, beam_width, "multi_rnn_cell")

      ## Train or eval
      if self.mode != contrib_learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = self.features["target_input"]
        target_output = self.features["target_output"]
        target_input = tf.transpose(target_input)
        target_output = tf.transpose(target_output)
        if self.length > 0:
          target_input = tf.slice(target_input, [0, 0], [self.length, -1])
          target_output = tf.slice(target_output, [0, 0], [self.length, -1])

        decoder_emb_inp = self._emb_lookup(
            self.embedding_decoder, target_input, is_decoder=True)

        seq_len = self.features["target_sequence_length"]
        padding = tf.transpose(
            tf.sequence_mask(seq_len, target_input.shape[0],
                             decoder_emb_inp.dtype))
        max_seq_len = tf.reduce_max(seq_len)
        o = decoder_emb_inp
        if self.mode == contrib_learn.ModeKeys.TRAIN:
          o = o * dropout(o.shape, o.dtype, 1.0 - hparams.dropout)
        inp = {"rnn": tf.einsum("tbf,fd->tbd", o, input_kernels[0])}
        new_states = build_rnn(theta[0], state0[0], inp, attention_cell,
                               attention_cell_grad, max_seq_len)
        attention_state = new_states["attention"]
        o = new_states["h"]
        for i in range(1, 4):
          c = tf.concat([o, attention_state], -1)
          if self.mode == contrib_learn.ModeKeys.TRAIN:
            c = c * dropout(c.shape, c.dtype, 1.0 - hparams.dropout)
          inp = {"rnn": tf.einsum("tbf,fd->tbd", c, input_kernels[i])}
          out = build_rnn(theta[i], state0[i], inp, lstm_cell, lstm_cell_grad,
                          max_seq_len)
          o = out["h"] + o if i > 1 else out["h"]

        out = tf.reshape(o * tf.expand_dims(padding, 2), [-1, self.num_units])

        logits = tf.matmul(
            tf.cast(out, self.output_layer.dtype), self.output_layer)
        label = tf.one_hot(
            tf.cast(tf.reshape(target_output, [-1]), tf.int32),
            self.tgt_vocab_size,
            1.0 - self.label_smoothing,
            self.label_smoothing / (self.tgt_vocab_size - 1),
            dtype=logits.dtype)
        loss = softmax_cross_entropy(logits, label)
        return tf.reduce_sum(loss), None, None

      ## Inference
      else:
        assert hparams.infer_mode == "beam_search"
        start_tokens = tf.fill([self.batch_size], hparams.tgt_sos_id)
        end_token = hparams.tgt_eos_id
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        coverage_penalty_weight = hparams.coverage_penalty_weight

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_infer_maximum_iterations(
            hparams, self.features["source_sequence_length"])

        def cell_fn(inputs, state):
          """Cell function used in decoder."""
          inp = {"rnn": tf.matmul(inputs, input_kernels[0])}
          atten_state, _ = attention_cell(theta[0], state[0], inp)
          o = atten_state["h"]
          new_states = [atten_state]
          for i in range(1, 4):
            ns, _ = lstm_cell(
                theta[i], state[i], {
                    "rnn":
                        tf.matmul(
                            tf.concat([o, atten_state["attention"]], -1),
                            input_kernels[i])
                })
            new_states.append(ns)
            if i > 1:
              o = ns["h"] + o
            else:
              o = ns["h"]
          return new_states, o

        my_decoder = beam_search_decoder.BeamSearchDecoder(
            cell=cell_fn,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=state0,
            beam_width=beam_width,
            output_layer=self.output_layer,
            max_tgt=maximum_iterations,
            length_penalty_weight=length_penalty_weight,
            coverage_penalty_weight=coverage_penalty_weight,
            dtype=self.dtype)

        # Dynamic decoding
        predicted_ids = decoder.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            swap_memory=True,
            scope=decoder_scope)

    return logits, decoder_cell_outputs, predicted_ids

  def _prepare_beam_search_decoder_inputs(self, beam_width, memory,
                                          source_sequence_length):
    memory = contrib_seq2seq.tile_batch(memory, multiplier=beam_width)
    source_sequence_length = contrib_seq2seq.tile_batch(
        source_sequence_length, multiplier=beam_width)
    batch_size = self.batch_size * beam_width
    return memory, source_sequence_length, batch_size

  def _build_encoder(self, hparams):
    """Build a GNMT encoder."""
    source = self.features["source"]
    source = tf.transpose(source)
    if self.length > 0:
      source = tf.slice(source, [0, 0], [self.length, -1])

    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      emb = tf.cast(
          self.encoder_emb_lookup_fn(self.embedding_encoder, source),
          self.dtype)
      seq_len = self.features["source_sequence_length"]
      padding = tf.transpose(
          tf.sequence_mask(seq_len, emb.shape[0], self.dtype))
      max_seq_len = tf.reduce_max(seq_len)
      if self.mode == contrib_learn.ModeKeys.TRAIN:
        emb = emb * dropout(emb.shape, emb.dtype, 1.0 - hparams.dropout)
      out = build_bid_rnn({"rnn": emb}, seq_len, hparams.num_units,
                          "bidirectional_rnn")
      out = out * tf.expand_dims(padding, 2)
      for i in range(3):
        orig_out = out
        if self.mode == contrib_learn.ModeKeys.TRAIN:
          out = out * dropout(out.shape, emb.dtype, 1.0 - hparams.dropout)
        inputs = {"rnn": out}
        o = build_uni_rnn(inputs, max_seq_len, hparams.num_units,
                          "rnn/uni_rnn_cell_%d" % i)
        if i > 0:
          o = o + orig_out
        out = o
      out = out * tf.expand_dims(padding, 2)

    return out
