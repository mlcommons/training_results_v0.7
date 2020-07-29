# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Several functions to initialize typical values of dataset parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.compat as tf

from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import attention
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import layers
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import optimizer
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import py_utils
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import rnn_cell
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import rnn_layers
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import schedule
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.tasks.mt import decoder
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.tasks.mt import encoder
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.tasks.mt import input_generator
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.tasks.mt import model
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import self_attention_layer


def InitTrainDatasetParams(vocab_size=None, params=None):
  """Initializes typical values for train datasets.

  Args:
    vocab_size: the number of tokens in your vocabulary. The default is None
      because this parameter is often not used.
    params: initial Params value, e.g. `NmtInput.Params()`.

  Returns:
    a `Params` object.
  """
  if params is None:
    params = input_generator.NmtInput.Params()
  params.is_nmt_example = True

  params.file_random_seed = 0

  # How many threads to run in parallel.
  params.file_parallelism = 16

  # Note, for training, we prefer to use big file_buffer_size (as long as all
  # fits in RAM), to more thoroughly randomize the training examples. when the
  # file_buffer_size too small, we run the risk of sequentially going over the
  # example as they are stored in the sstable which may not be random (e.g.
  # maybe alphabetically ordered).
  params.file_buffer_size = 10000000

  if vocab_size is not None:
    params.tokenizer.vocab_size = vocab_size

  # The bucket upper bound is determined based on an exponentially growing
  # scheme, with _GenerateBuckets(10, 100) resulting buckets starting from
  # minimum bucket size of 10 to maximum bucket size of 137.
  # For word and sub-word level NMT, we train on sequences up to maximum
  # bucket size and discard the examples that are longer than 137.
  params.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137]

  # The bucket batch limit determines how many examples are there in each
  # batch during training. We reduce the batch size for the buckets that
  # have higher upper bound (batches that consist of longer sequences eg.
  # 98, 137) in order to prevent out of memory issues.
  # Note that this hyperparameter varies widely based on the model and language.
  # larger models may warrant smaller batches in order to fit in memory, for
  # example; and ideographical languages like Chinese may benefit from more
  # buckets.
  params.bucket_batch_limit = [128] * 8 + [64]
  return params


def InitTestDatasetParams(vocab_size=None, params=None):
  """Initializes typical values for test and dev datasets.

  Args:
    vocab_size: the number of tokens in your vocabulary.
    params: initial Params value, e.g. `NmtInput.Params()`.

  Returns:
    a `Params` object.
  """

  if params is None:
    params = input_generator.NmtInput.Params()

  params.file_random_seed = 27182818

  # How many threads to run in parallel.
  params.file_parallelism = 1

  # In order to make exactly one pass over the dev/test sets, we set buffer
  # size to 1. Greater numbers may cause inaccurate dev/test scores.
  params.file_buffer_size = 1

  if vocab_size is not None:
    params.tokenizer.vocab_size = vocab_size

  # The largest bucket upper bound must be larger than the longest sequence
  # length in dev/test set. Since we discard sequences longer than the
  # max(bucket_upper_bound) we may end up having scores based on only shorter
  # sequences only if we mistakenly set this to be too small.
  params.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
  params.bucket_batch_limit = [128] * 8 + [64] + [32]
  return params


def InitTransformerTestBuckets(params):
  params.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
  params.bucket_batch_limit = [16] * 10
  return params


def InitTransformerTrainBuckets(params):
  params.bucket_upper_bound = [8, 12, 16, 24, 32, 48, 64, 96]
  params.bucket_batch_limit = [512, 341, 256, 170, 128, 85, 64, 42]
  return params


def SetupTransformerParams(p,
                           name,
                           vocab_size,
                           model_dim,
                           hidden_dim,
                           num_heads,
                           num_layers,
                           learning_rate,
                           warmup_steps,
                           residual_dropout_prob=0.1,
                           input_dropout_prob=0.0,
                           atten_dropout_prob=0.0,
                           relu_dropout_prob=0.0,
                           label_smoothing_uncertainty=0.1,
                           is_transparent=False,
                           activation='RELU',
                           add_unnormalized_residuals=False,
                           atten_hidden_dim=0,
                           num_encoder_layers=None,
                           num_decoder_layers=None):
  """Common model setup for different transformer models.

  Args:
    p: The initial params object to modify.
    name: An identifier for an instance of a transformer model.
    vocab_size: an integer representing the size of the vocabulary, probably
         16000 or 32000.
    model_dim: dimension of the transformer block (column)
    hidden_dim: dimension of Feed-Forward neural network in each layer
    num_heads: number of attention heads to use for the transformer
    num_layers: number of layers in the transformer
    learning_rate: learning rate for Adam. For the base model, we use 1.0; for
         the big model, 3.0
    warmup_steps: warmup steps for `TransformerSchedule`. For the
         base model, we use 4000; for the big model, 40000
    residual_dropout_prob: dropout prob to the output of each sub-layer before
         it is added to the sub-layer input
    input_dropout_prob: dropout prob to the sums of the token embeddings and the
         position embeddings
    atten_dropout_prob: dropout prob to the attention weights in each
         Transformer attention sub-layer
    relu_dropout_prob: dropout prob to the inner layer output (ReLU activation)
         in each Transformer feed-forward sub-layer
    label_smoothing_uncertainty: if this value is 0, no label smoothing will be
         applied
    is_transparent: If set, decoder layers attend to weighted combinations of
        encoder layers.
    activation: Non-linearity for feed-forward layers.
    add_unnormalized_residuals: If set, uses un-normalized residuals in
        TransformerAttentionLayer
    atten_hidden_dim: Explicitly set attention hidden dim.
    num_encoder_layers: to set a different number of layers for the encoder.
    num_decoder_layers: to set a different number of layers for the decoder.

  Returns:
    A Params object containing the parameters that specify a transformer model
    (Vaswani 2017)

  """
  p.name = name

  # Transformer encoder and decoder setup
  num_encoder_layers = num_encoder_layers or num_layers
  num_decoder_layers = num_decoder_layers or num_layers
  p.encoder = SetupTransformerEncoder(
      model_dim, vocab_size, num_encoder_layers, num_heads, hidden_dim,
      residual_dropout_prob, input_dropout_prob, atten_dropout_prob,
      relu_dropout_prob, is_transparent, activation, add_unnormalized_residuals,
      atten_hidden_dim)
  p.decoder = SetupTransformerDecoder(
      model_dim, vocab_size, num_decoder_layers, num_heads, hidden_dim,
      residual_dropout_prob, input_dropout_prob, atten_dropout_prob,
      relu_dropout_prob, label_smoothing_uncertainty, is_transparent,
      activation, add_unnormalized_residuals, atten_hidden_dim)

  p.train.Set(
      learning_rate=learning_rate,
      optimizer=optimizer.Adam.ParamsB(),
      clip_gradient_norm_to_value=0.0,
      grad_norm_to_clip_to_zero=0.0,
      lr_schedule=schedule.TransformerSchedule.Params().Set(
          warmup_steps=warmup_steps, worker_replicas=1, model_dim=model_dim))

  p.eval.samples_per_summary = 12000
  return p


def SetupTransformerDecoder(model_dim,
                            vocab_size,
                            num_layers,
                            num_heads,
                            hidden_dim,
                            residual_dropout_prob=0.1,
                            input_dropout_prob=0.0,
                            atten_dropout_prob=0.0,
                            relu_dropout_prob=0.0,
                            label_smoothing_uncertainty=0.1,
                            is_transparent=False,
                            activation='RELU',
                            add_unnormalized_residuals=False,
                            atten_hidden_dim=0):
  """Common setup for transformer model decoder."""
  disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
  default_params_init = py_utils.WeightInit.Xavier(1.0)
  emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

  # Decoder
  decoder_params = decoder.TransformerDecoder.Params()
  decoder_params.source_dim = model_dim
  decoder_params.model_dim = model_dim
  decoder_params.num_trans_layers = num_layers
  decoder_params.input_dropout_prob = input_dropout_prob

  decoder_params.token_emb.Set(
      vocab_size=vocab_size,
      embedding_dim=model_dim,
      max_num_shards=16,
      params_init=emb_params_init,
      vn=disable_vn,
      scale_sqrt_depth=True)

  decoder_params.position_emb.Set(
      embedding_dim=model_dim, trainable_scaling=False, vn=disable_vn)

  decoder_params.trans_tpl.source_dim = model_dim
  decoder_params.trans_tpl.tr_atten_tpl.Set(
      source_dim=model_dim,
      num_attention_heads=num_heads,
      residual_dropout_prob=residual_dropout_prob,
      atten_dropout_prob=atten_dropout_prob,
      params_init=default_params_init,
      add_unnormalized_input=add_unnormalized_residuals,
      atten_hidden_dim=atten_hidden_dim,
      vn=disable_vn)

  decoder_params.trans_tpl.tr_atten_tpl.atten_tpl.Set(
      enable_ctx_pre_proj=True,
      enable_ctx_post_proj=True,
      context_dim=model_dim,
      vn=disable_vn)

  decoder_params.trans_tpl.tr_fflayer_tpl.Set(
      input_dim=model_dim,
      hidden_dim=hidden_dim,
      residual_dropout_prob=residual_dropout_prob,
      relu_dropout_prob=relu_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn,
      activation=activation)

  decoder_params.softmax.Set(
      num_classes=vocab_size,
      vn=disable_vn,
      params_init=emb_params_init,
      num_shards=16)

  decoder_params.per_word_avg_loss = True
  decoder_params.label_smoothing = layers.UniformLabelSmoother.Params()
  decoder_params.label_smoothing.num_classes = vocab_size
  decoder_params.label_smoothing.uncertainty = label_smoothing_uncertainty

  if is_transparent:
    decoder_params.is_transparent = True

  return decoder_params


def SetupTransformerEncoder(model_dim,
                            vocab_size,
                            num_layers,
                            num_heads,
                            hidden_dim,
                            residual_dropout_prob=0.1,
                            input_dropout_prob=0.0,
                            atten_dropout_prob=0.0,
                            relu_dropout_prob=0.0,
                            is_transparent=False,
                            activation='RELU',
                            add_unnormalized_residuals=False,
                            atten_hidden_dim=0):
  """Common setup for transformer model encoder.

  Args:
   model_dim: specifies dimension of transformer layers, token embeddings,
    and positional embeddings as well context vectors (attention values).
   vocab_size: for token embeddings.
   num_layers: number of transformer layers.
   num_heads: number of attention heads.
   hidden_dim: in transformer feedforward layer.
   residual_dropout_prob: used in transformer feedforward and attention layer.
   input_dropout_prob: input dropout.
   atten_dropout_prob: used in attention layer.
   relu_dropout_prob: used in transformer feedforward layer.
   is_transparent: if set, outputs a merger of embeddings and layer outputs.
   activation: Non-linearity for feed-forward layers.
   add_unnormalized_residuals: If set, uses un-normalized residuals in
     TransformerAttentionLayer
   atten_hidden_dim: Explicitly set attention hidden dim.

  Returns:
   Encoder params.
  """
  disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
  default_params_init = py_utils.WeightInit.Xavier(1.0)
  emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

  # Encoder
  encoder_params = encoder.TransformerEncoder.Params()

  encoder_params.token_emb.Set(
      embedding_dim=model_dim,
      max_num_shards=16,
      params_init=emb_params_init,
      vocab_size=vocab_size,
      vn=disable_vn,
      scale_sqrt_depth=True)

  encoder_params.position_emb.Set(
      embedding_dim=model_dim, trainable_scaling=False, vn=disable_vn)

  # Encoder TransformerStack params
  encoder_params.model_dim = model_dim
  encoder_params.transformer_stack.model_dim = model_dim
  encoder_params.transformer_stack.num_transformer_layers = num_layers
  encoder_params.input_dropout_prob = input_dropout_prob

  encoder_params.transformer_stack.transformer_tpl.tr_atten_tpl.Set(
      num_attention_heads=num_heads,
      residual_dropout_prob=residual_dropout_prob,
      atten_dropout_prob=atten_dropout_prob,
      params_init=default_params_init,
      add_unnormalized_input=add_unnormalized_residuals,
      atten_hidden_dim=atten_hidden_dim,
      vn=disable_vn)

  encoder_params.transformer_stack.transformer_tpl.tr_atten_tpl.atten_tpl.Set(
      num_attention_heads=num_heads,
      enable_ctx_pre_proj=True,
      enable_ctx_post_proj=True,
      context_dim=model_dim,
      vn=disable_vn)

  encoder_params.transformer_stack.transformer_tpl.tr_fflayer_tpl.Set(
      hidden_dim=hidden_dim,
      residual_dropout_prob=residual_dropout_prob,
      relu_dropout_prob=relu_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn,
      activation=activation)

  if is_transparent:
    encoder_params.transformer_stack.is_transparent = True

  return encoder_params


def SetupRNMTParams(p,
                    name,
                    vocab_size,
                    embedding_dim,
                    hidden_dim,
                    num_heads,
                    num_encoder_layers,
                    num_decoder_layers,
                    learning_rate,
                    lr_warmup_steps,
                    lr_decay_start,
                    lr_decay_end,
                    lr_min,
                    atten_dropout_prob,
                    residual_dropout_prob,
                    ls_uncertainty,
                    l2_regularizer_weight,
                    is_transparent=False,
                    num_hyps_per_beam=16,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    adam_epsilon=8e-07):
  """Creates RNMT+ params common to all datasets.

  Args:
    p: The initial params object to modify.
    name: A descriptive name for your model.
    vocab_size: size of the vocabulary. Probably 32000 or 16000.
    embedding_dim: Dimension of token embeddings.
    hidden_dim: LSTM cell size.
    num_heads: number of attention heads.
    num_encoder_layers: Number of layers in the encoder.
    num_decoder_layers: Number of layers in the decoder.
    learning_rate: Optimizer learning rate.
    lr_warmup_steps: Warm-up steps for the optimizer.
    lr_decay_start: Learning rate exponential decay starting step.
    lr_decay_end: Learning rate exponential decay end step.
    lr_min: Minimum learning rate (ratio with initial learning rate).
    atten_dropout_prob: Dropout for the attention.
    residual_dropout_prob: Dropout for residual layers.
    ls_uncertainty: Label smoothing uncertainty.
    l2_regularizer_weight: Weight for l2 regularization on parameters.
    is_transparent: If set, decoder attends to weighted combination of encoder
      layers.
    num_hyps_per_beam: Number of hyps to keep per source sequence.
    adam_beta1: Beta-1 parameter of Adam optimizer.
    adam_beta2: Beta-2 parameter of Adam optimizer.
    adam_epsilon: Epsilon parameter of Adam optimizer.

  Returns:
    a Params() object specifying the RNMT+ Parameters.
  """

  # TODO(orhanf): add transparent connections.
  del is_transparent

  p.name = name

  default_params_init = py_utils.WeightInit.Uniform(0.04)
  rnn_cell_tpl = rnn_cell.LayerNormalizedLSTMCellSimple.Params().Set(
      num_output_nodes=hidden_dim,
      output_nonlinearity=False,
      params_init=default_params_init)

  # RNMT+ encoder setup.
  p.encoder = encoder.MTEncoderBiRNN.Params().Set(
      num_lstm_layers=num_encoder_layers,
      lstm_cell_size=hidden_dim,
      encoder_out_dim=hidden_dim,
      lstm_tpl=rnn_cell_tpl.Copy(),
      dropout_prob=residual_dropout_prob)
  p.encoder.emb.embedding_dim = embedding_dim
  p.encoder.emb.vocab_size = vocab_size

  # RNMT+ decoder setup.
  p.decoder = decoder.MTDecoderV1.Params().Set(
      rnn_layers=num_decoder_layers,
      rnn_cell_tpl=rnn_cell_tpl.Copy(),
      atten_rnn_cell_tpl=rnn_cell_tpl.Copy(),
      dropout_prob=residual_dropout_prob,
      attention=attention.MultiHeadedAttention.Params().Set(
          source_dim=hidden_dim,
          hidden_dim=hidden_dim,
          query_dim=hidden_dim,
          context_dim=hidden_dim,
          num_attention_heads=num_heads,
          inner_atten_params=attention.AdditiveAttention.Params(),
          use_source_vec_as_attention_value=True,
          enable_ctx_pre_proj=False,
          enable_query_proj=True,
          atten_dropout_prob=atten_dropout_prob,
          atten_dropout_deterministic=True),
      atten_rnn_cls=rnn_layers.FRNNWithAttention,
      feed_attention_context_vec_to_softmax=True,
      label_smoothing=layers.UniformLabelSmoother.Params().Set(
          num_classes=vocab_size, uncertainty=ls_uncertainty))
  p.decoder.emb.vocab_size = vocab_size
  p.decoder.emb.embedding_dim = embedding_dim
  p.decoder.softmax.num_classes = vocab_size
  p.decoder.source_dim = hidden_dim

  # Inference related.
  p.decoder.beam_search.num_hyps_per_beam = num_hyps_per_beam

  # Optimization setup.
  learning_rate_schedule = (
      schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
      .Set(
          warmup=lr_warmup_steps,
          decay_start=lr_decay_start,
          decay_end=lr_decay_end,
          min=lr_min))
  p.train.Set(
      l2_regularizer_weight=l2_regularizer_weight,
      grad_norm_tracker=layers.GradNormTracker.Params().Set(
          name='gradient_norm_tracker'),
      learning_rate=learning_rate,
      lr_schedule=learning_rate_schedule,
      grad_norm_to_clip_to_zero=100000.0,
      optimizer=optimizer.Adam.Params().Set(
          beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon),
  )

  # Evaluation related
  p.eval.samples_per_summary = 12000
  return p


def SetupTransformerBatchMajorParams(name,
                                     vocab_size,
                                     model_dim,
                                     hidden_dim,
                                     num_heads,
                                     num_layers,
                                     learning_rate,
                                     warmup_steps,
                                     inference_source_language,
                                     inference_target_language,
                                     residual_dropout_prob=0.1,
                                     input_dropout_prob=0.0,
                                     atten_dropout_prob=0.0,
                                     relu_dropout_prob=0.0,
                                     label_smoothing_uncertainty=0.1,
                                     activation='RELU',
                                     add_unnormalized_residuals=False,
                                     atten_hidden_dim=0,
                                     encoder_version='V1',
                                     decoder_version='V1',
                                     packed_input=False,
                                     use_fast_projection_layer=False,
                                     enable_per_dim_scale=True,
                                     use_fused_layernorm=False,
                                     use_bf16_activations=False,
                                     use_bias=True,
                                     xla_num_partitions=None):
  """Common model setup for Transformer batch-major model.

  Args:
    name: A string, the identifier for an instance of a Transformer model.
    vocab_size: An integer, representing the size of the vocabulary, probably
      16000 or 32000.
    model_dim: An integer, representing the dimension of the Transformer block.
    hidden_dim: An integer, representing the dimension of Feed-Forward neural
      network in each layer.
    num_heads: An integer, representing the number of attention heads to use for
      the Transformer.
    num_layers: An integer, representing the number of layers in the
      Transformer.
    learning_rate: A float, representing the learning rate for Adam. For the
      base model, we use 1.0; for the big model, 3.0.
    warmup_steps: An integer, representing the Warmup steps for
      TransformerSchedule. For the base model, we use 4000; for the big model,
      40000.
    inference_source_language: A string, the source language to use in
      inference.
    inference_target_language: A string, the target language to use in
      inference.
    residual_dropout_prob: A float, representing the dropout prob to the output
      of each sub-layer before it is added to the sub-layer input.
    input_dropout_prob: A float, representing the dropout prob to the sums of
      the token embeddings and the position embeddings.
    atten_dropout_prob: A float, representing the dropout prob to the attention
      weights in each Transformer attention sub-layer.
    relu_dropout_prob: A float, representing the dropout prob to the inner layer
      output (ReLU activation) in each Transformer feed-forward sub-layer.
    label_smoothing_uncertainty: A float, representing the uncertainty in label
      smoothing. If this value is 0, no label smoothing will be applied.
    activation: A string, the non-linearity method used for feed-forward layers.
      This parameter only affects decoder.
    add_unnormalized_residuals: A bool. If set, uses un-normalized residuals in
      Transformer attention layers.
    atten_hidden_dim: An integer, representing the hidden dim of the attention.
      If 0, default is model_dim. This parameter only affects decoder.
    encoder_version: The encoder to use. Allowed options are "V1" and "V2".
    decoder_version: The decoder to use. Allowed options are "V1" and "V2".
    packed_input: Whether to enable packed input.
    use_fast_projection_layer: Whether to use fast projection layer to remove
      the data formatting overheads.
    enable_per_dim_scale: Whether to enable per_dim_scale.
    use_fused_layernorm: Whether to use fused layer normalization.
    use_bf16_activations: Whether to use bfloat16 for activations.
    use_bias: Whether to use bias for projection layer.

  Returns:
    A Params object containing the parameters that specify a Transformer model.
  """
  p = model.TransformerBatchMajorModel.Params()
  p.name = name

  # Transformer encoder and decoder setup
  if encoder_version == 'V1':
    encoder_params_fn = SetupTransformerBatchMajorEncoderV1
  elif encoder_version == 'V2':
    encoder_params_fn = SetupTransformerBatchMajorEncoderV2
  else:
    raise NotImplementedError('encoder_version: "%s" not recognized.' %
                              encoder_version)
  p.encoder = encoder_params_fn(
      model_dim,
      vocab_size,
      num_layers,
      num_heads,
      hidden_dim,
      residual_dropout_prob,
      input_dropout_prob,
      atten_dropout_prob,
      relu_dropout_prob,
      activation,
      add_unnormalized_residuals,
      packed_input=packed_input,
      use_fast_projection_layer=use_fast_projection_layer,
      enable_per_dim_scale=enable_per_dim_scale,
      use_fused_layernorm=use_fused_layernorm,
      use_bf16_activations=use_bf16_activations,
      use_bias=use_bias,
      xla_num_partitions=xla_num_partitions)

  if decoder_version == 'V1':
    decoder_params_fn = SetupTransformerBatchMajorDecoderV1
  elif decoder_version == 'V2':
    decoder_params_fn = SetupTransformerBatchMajorDecoderV2
  else:
    raise NotImplementedError('decoder_version: "%s" not recognized.' %
                              decoder_version)
  p.decoder = decoder_params_fn(
      model_dim,
      vocab_size,
      num_layers,
      num_heads,
      hidden_dim,
      residual_dropout_prob,
      input_dropout_prob,
      atten_dropout_prob,
      relu_dropout_prob,
      label_smoothing_uncertainty,
      activation,
      add_unnormalized_residuals,
      atten_hidden_dim,
      packed_input=packed_input,
      use_fast_projection_layer=use_fast_projection_layer,
      enable_per_dim_scale=enable_per_dim_scale,
      use_fused_layernorm=use_fused_layernorm,
      use_bias=use_bias)

  p.train.Set(
      learning_rate=learning_rate,
      optimizer=optimizer.Adam.ParamsB(),
      clip_gradient_norm_to_value=0.0,
      grad_norm_to_clip_to_zero=0.0,
      lr_schedule=schedule.TransformerSchedule.Params().Set(
          warmup_steps=warmup_steps, worker_replicas=1, model_dim=model_dim))

  #p.eval.inference_source_language = inference_source_language
  #p.eval.inference_target_language = inference_target_language
  p.eval.samples_per_summary = 12000

  return p


def SetupTransformerBatchMajorDecoderV1(model_dim,
                                        vocab_size,
                                        num_layers,
                                        num_heads,
                                        hidden_dim,
                                        residual_dropout_prob=0.1,
                                        input_dropout_prob=0.0,
                                        atten_dropout_prob=0.0,
                                        relu_dropout_prob=0.0,
                                        label_smoothing_uncertainty=0.1,
                                        activation='RELU',
                                        add_unnormalized_residuals=False,
                                        atten_hidden_dim=0,
                                        packed_input=False,
                                        use_fast_projection_layer=False,
                                        enable_per_dim_scale=True,
                                        use_fused_layernorm=False,
                                        use_bias=True):
  """Common setup for batch major transformer model decoder.

  Args:
   model_dim: specifies dimension of transformer layers, token embeddings, and
     positional embeddings as well context vectors (attention values).
   vocab_size: for token embeddings.
   num_layers: number of transformer layers.
   num_heads: number of attention heads.
   hidden_dim: in transformer feedforward layer.
   residual_dropout_prob: used in transformer feedforward and attention layer.
   input_dropout_prob: input dropout.
   atten_dropout_prob: used in attention layer.
   relu_dropout_prob: used in transformer feedforward layer.
   label_smoothing_uncertainty: A float, representing the uncertainty in label
     smoothing. If this value is 0, no label smoothing will be applied.
   activation: Non-linearity for feed-forward layers.
   add_unnormalized_residuals: If set, uses un-normalized residuals in
     TransformerAttentionLayer
   atten_hidden_dim: Explicitly set attention hidden dim. If 0, default is
     model_dim.
   packed_input: Whether to enable packed input.
   use_fast_projection_layer: Whether to use fast projection layer to remove the
     data formatting overheads.
   enable_per_dim_scale: Whether to enable per_dim_scale.
   use_fused_layernorm: Whether to use fused layer normalization.
   use_bias: Whether to use bias for projection layer.

  Returns:
   Decoder params.
  """
  disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
  default_params_init = py_utils.WeightInit.Xavier(1.0)
  emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

  # Decoder
  decoder_params = decoder.TransformerBatchMajorDecoder.Params()

  decoder_params.source_dim = model_dim
  decoder_params.model_dim = model_dim
  decoder_params.num_trans_layers = num_layers
  decoder_params.input_dropout_prob = input_dropout_prob
  decoder_params.packed_input = packed_input
  decoder_params.use_fused_layernorm = use_fused_layernorm

  decoder_params.token_emb.Set(
      vocab_size=vocab_size,
      embedding_dim=model_dim,
      max_num_shards=16,
      params_init=emb_params_init,
      vn=disable_vn,
      scale_sqrt_depth=True)

  decoder_params.position_emb.Set(
      embedding_dim=model_dim, trainable_scaling=False, vn=disable_vn)

  decoder_params.trans_decoder_tpl.packed_input = packed_input
  decoder_params.trans_decoder_tpl.input_dim = model_dim
  decoder_params.trans_decoder_tpl.tr_atten_tpl.Set(
      input_dim=model_dim,
      num_heads=num_heads,
      residual_dropout_prob=residual_dropout_prob,
      atten_dropout_prob=atten_dropout_prob,
      params_init=default_params_init,
      add_unnormalized_input=add_unnormalized_residuals,
      hidden_dim=atten_hidden_dim,
      vn=disable_vn)

  decoder_params.trans_decoder_tpl.tr_atten_tpl.ln_tpl.Set(
      use_fused_layernorm=use_fused_layernorm)
  decoder_params.trans_decoder_tpl.tr_atten_tpl.atten_tpl.Set(
      vn=disable_vn,
      packed_input=packed_input,
      enable_per_dim_scale=enable_per_dim_scale,
      use_bias=use_bias)
  decoder_params.trans_decoder_tpl.tr_fflayer_tpl.Set(
      input_dim=model_dim,
      hidden_dim=hidden_dim,
      residual_dropout_prob=residual_dropout_prob,
      relu_dropout_prob=relu_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn,
      activation=activation)
  decoder_params.trans_decoder_tpl.tr_fflayer_tpl.ln_tpl.Set(
      use_fused_layernorm=use_fused_layernorm)
  decoder_params.trans_decoder_tpl.tr_fflayer_tpl.fflayer_tpl.projection.Set(
      use_einsum=use_fast_projection_layer)

  if add_unnormalized_residuals:
    decoder_params.final_layer_norm = True

  decoder_params.softmax.Set(
      num_classes=vocab_size,
      vn=disable_vn,
      params_init=emb_params_init,
      num_shards=16)

  decoder_params.per_word_avg_loss = True
  decoder_params.label_smoothing = layers.UniformLabelSmoother.Params()
  decoder_params.label_smoothing.num_classes = vocab_size
  decoder_params.label_smoothing.uncertainty = label_smoothing_uncertainty

  return decoder_params


def SetupTransformerBatchMajorEncoderV1(model_dim,
                                        vocab_size,
                                        num_layers,
                                        num_heads,
                                        hidden_dim,
                                        residual_dropout_prob=0.1,
                                        input_dropout_prob=0.0,
                                        atten_dropout_prob=0.0,
                                        relu_dropout_prob=0.0,
                                        activation='RELU',
                                        add_unnormalized_residuals=False,
                                        packed_input=False,
                                        use_fast_projection_layer=False,
                                        enable_per_dim_scale=True,
                                        use_fused_layernorm=False,
                                        use_bf16_activations=False,
                                        use_bias=True,
                                        xla_num_partitions=None):
  """Common setup for transformer batch major model encoder.

  Args:
   model_dim: specifies dimension of transformer layers, token embeddings, and
     positional embeddings as well context vectors (attention values).
   vocab_size: for token embeddings.
   num_layers: number of transformer layers.
   num_heads: number of attention heads.
   hidden_dim: in transformer feedforward layer.
   residual_dropout_prob: used in transformer feedforward and attention layer.
   input_dropout_prob: input dropout.
   atten_dropout_prob: used in attention layer.
   relu_dropout_prob: used in transformer feedforward layer.
   activation: Non-linearity for feed-forward layers. (Unused)
   add_unnormalized_residuals: If set, uses un-normalized residuals in
     TransformerAttentionLayer
   packed_input: Whether to enable packed input.
   use_fast_projection_layer: Whether to use fast projection layer to remove the
     data formatting overheads.
   enable_per_dim_scale: Whether to enable per_dim_scale.
   use_fused_layernorm: Whether to use fused layer normalization.
   use_bf16_activations: Whether to use bfloat16 for activations.
   use_bias: Whether to use bias for projection layer.

  Returns:
   Encoder params.
  """
  # TODO(shibow) Add 'GLEU' activation option for batch major encoder.
  # GELU should be used to reduce quality loss when casting weights from fp32
  # to bf16 for inference (b/123993529).
  del activation
  # EncoderV1 has already had fast projection layer as default.
  del use_fast_projection_layer

  disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
  default_params_init = py_utils.WeightInit.Xavier(1.0)
  emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

  # Encoder
  encoder_params = encoder.TransformerBatchMajorEncoder.Params()
  encoder_params.params_init = default_params_init
  encoder_params.model_dim = model_dim
  encoder_params.input_dropout_prob = input_dropout_prob
  encoder_params.packed_input = packed_input
  encoder_params.use_fused_layernorm = use_fused_layernorm

  encoder_params.token_emb.Set(
      embedding_dim=model_dim,
      max_num_shards=16,
      params_init=emb_params_init,
      vocab_size=vocab_size,
      vn=disable_vn,
      scale_sqrt_depth=True)

  encoder_params.position_emb.Set(
      embedding_dim=model_dim, trainable_scaling=False, vn=disable_vn)

  builder_params = self_attention_layer.Builder.Params().Set(
      model_dim=model_dim,
      num_heads=num_heads,
      ff_hidden_dim=hidden_dim,
      residual_dropout_prob=residual_dropout_prob,
      relu_dropout_prob=relu_dropout_prob,
      atten_dropout_prob=atten_dropout_prob,
      selfatten_add_unnormalized_input=add_unnormalized_residuals,
      selfatten_enable_value_proj=True,
      packed_input=packed_input,
      enable_per_dim_scale=enable_per_dim_scale,
      use_fused_layernorm=use_fused_layernorm,
      fprop_dtype=tf.bfloat16 if use_bf16_activations else None,
      use_bias=use_bias,
      xla_num_partitions=xla_num_partitions)
  stack = (
      builder_params.Instantiate().TransformerStack(
          name='transformer_stack', num_layers=num_layers))
  encoder_params.transformer_stack = (
      self_attention_layer.StackedTransformerEncoderLayers.Cast(stack))
  if add_unnormalized_residuals:
    encoder_params.final_layer_norm = True

  return encoder_params
