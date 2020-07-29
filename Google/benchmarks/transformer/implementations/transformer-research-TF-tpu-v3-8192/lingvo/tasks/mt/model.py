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
"""MT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.compat as tf
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import base_layer
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import base_model
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import insertion
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import metrics
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import ml_perf_bleu_metric
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import py_utils
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.tasks.mt import decoder
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.tasks.mt import encoder
from six.moves import range


class MTBaseModel(base_model.BaseTask):
  """Base Class for NMT models."""

  def _EncoderDevice(self):
    """Returns the device to run the encoder computation."""
    return tf.device('')

  def _DecoderDevice(self):
    """Returns the device to run the decoder computation."""
    return tf.device('')

  @base_layer.initializer
  def __init__(self, params):
    super(MTBaseModel, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      with self._EncoderDevice():
        if p.encoder:
          self.CreateChild('enc', p.encoder)
      with self._DecoderDevice():
        self.CreateChild('dec', p.decoder)

  def ComputePredictions(self, theta, batch):
    p = self.params

    with self._EncoderDevice():
      encoder_outputs = (
          self.enc.FProp(theta.enc, batch.src) if p.encoder else None)
    with self._DecoderDevice():
      predictions = self.dec.ComputePredictions(theta.dec, encoder_outputs,
                                                batch.tgt)
      if isinstance(predictions, py_utils.NestedMap):
        # Pass through encoder output as well for possible use as a FProp output
        # for various meta-MT modeling approaches, such as MT quality estimation
        # classification.
        predictions['encoder_outputs'] = encoder_outputs
      return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    with self._DecoderDevice():
      return self.dec.ComputeLoss(theta.dec, predictions, input_batch.tgt)

  def _GetTokenizerKeyToUse(self, key):
    """Returns a tokenizer key to use for the provided `key`."""
    if key in self.input_generator.tokenizer_dict:
      return key
    return None


  def _BeamSearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      encoder_outputs = self.dec.AddExtraDecodingInfo(encoder_outputs,
                                                      input_batch.tgt)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)
      return self._ProcessBeamSearchDecodeOut(input_batch, encoder_outputs,
                                              decoder_outs)

  def _ProcessBeamSearchDecodeOut(self, input_batch, encoder_outputs,
                                  decoder_outs):
    self.r1_shape = decoder_outs[0]
    self.r2_shape = decoder_outs[1]
    self.r3_shape = decoder_outs[2]
    tf.logging.info('r1_shape: %s', self.r1_shape)
    tf.logging.info('r2_shape: %s', self.r2_shape)
    tf.logging.info('r3_shape: %s', self.r3_shape)

    hyps = decoder_outs[3]
    prev_hyps = decoder_outs[4]
    done_hyps = decoder_outs[5]
    scores = decoder_outs[6]
    atten_probs = decoder_outs[7]
    eos_scores = decoder_outs[8]
    eos_atten_probs = decoder_outs[9]
    source_seq_lengths = decoder_outs[10]

    tlen = tf.cast(
        tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
        tf.int32)
    ret_dict = {
        'target_ids': input_batch.tgt.ids[:, 1:],
        'eval_weight': input_batch.eval_weight,
        'tlen': tlen,
        'hyps': hyps,
        'prev_hyps': prev_hyps,
        'done_hyps': done_hyps,
        'scores': scores,
        'atten_probs': atten_probs,
        'eos_scores': eos_scores,
        'eos_atten_probs': eos_atten_probs,
        'source_seq_lengths': source_seq_lengths,
    }
    return ret_dict

  def PostProcessDecodeHost(self, metrics_dict):
    p = self.params
    ret_dict = {
        'target_ids': metrics_dict['target_ids'],
        'eval_weight': metrics_dict['eval_weight'],
        'tlen': metrics_dict['tlen'],
    }
    r1_shape = self.r1_shape
    r2_shape = self.r2_shape
    r3_shape = self.r3_shape
    ret_ids = self.dec.BeamSearchDecodePostProcess(
        p.decoder.beam_search.num_hyps_per_beam,
        p.decoder.target_seq_len,
        r1_shape,
        r2_shape,
        r3_shape,
        metrics_dict['hyps'],
        metrics_dict['prev_hyps'],
        metrics_dict['done_hyps'],
        metrics_dict['scores'],
        metrics_dict['atten_probs'],
        metrics_dict['eos_scores'],
        metrics_dict['eos_atten_probs'],
        metrics_dict['source_seq_lengths'],
        [])
    (final_done_hyps, topk_hyps, topk_ids, topk_lens,
     topk_scores) = ret_ids[0:5]
    ret_dict['top_ids'] = topk_ids[::p.decoder.beam_search.num_hyps_per_beam]
    ret_dict['top_lens'] = topk_lens[::p.decoder.beam_search.num_hyps_per_beam]
    return ret_dict

  def _PostProcessBeamSearchDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post processes the output from `_BeamSearchDecode`."""
    tgt_ids = dec_out_dict['target_ids']
    tlen = dec_out_dict['tlen']
    top_ids = dec_out_dict['top_ids']
    top_lens = dec_out_dict['top_lens']
    eval_weight = dec_out_dict['eval_weight']

    total_eval_weight = 0.0

    targets = self.input_generator.PythonIdsToStrings(tgt_ids, tlen)
    num_samples = len(targets)
    tf.logging.info('num_samples: %d', num_samples)
    top_decoded = self.input_generator.PythonIdsToStrings(top_ids, top_lens - 1)
    assert num_samples == len(top_decoded), ('%s vs %s' %
                                             (num_samples, len(top_decoded)))
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)
    for i in range(num_samples):
      tgt = targets[i]
      top_hyp = top_decoded[i]
      example_eval_weight = eval_weight[i]
      total_eval_weight += example_eval_weight
      dec_metrics_dict['ml_perf_bleu'].Update(tgt, top_hyp, example_eval_weight)
    tf.logging.info('total_eval_weight: %f', total_eval_weight)
    return []

  def CreateDecoderMetrics(self):
    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'ml_perf_bleu': ml_perf_bleu_metric.MlPerfBleuMetric(),
    }
    return decoder_metrics

  def Decode(self, input_batch):
    """Constructs the decoding graph."""
    return self._BeamSearchDecode(input_batch)

  def PostProcessDecodeOut(self, dec_out, dec_metrics):
    return self._PostProcessBeamSearchDecodeOut(dec_out, dec_metrics)


class TransformerModel(MTBaseModel):
  """Transformer Model.

  Implements Attention is All You Need:
  https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super(TransformerModel, cls).Params()
    p.encoder = encoder.TransformerEncoder.Params()
    p.decoder = decoder.TransformerDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerModel, self).__init__(params)
    p = self.params
    assert p.encoder.model_dim == p.decoder.source_dim


class RNMTModel(MTBaseModel):
  """RNMT+ Model.

  Implements RNMT Variants in The Best of Both Worlds paper:
  https://aclweb.org/anthology/P18-1008
  """

  @classmethod
  def Params(cls):
    p = super(RNMTModel, cls).Params()
    p.encoder = encoder.MTEncoderBiRNN.Params()
    p.decoder = decoder.MTDecoderV1.Params()
    return p


class InsertionModel(MTBaseModel):
  """Insertion-based model.

  References:
    KERMIT: https://arxiv.org/pdf/1906.01604.pdf
    Insertion Transformer: https://arxiv.org/pdf/1902.03249.pdf
  """

  @classmethod
  def Params(cls):
    p = super(InsertionModel, cls).Params()
    p.decoder = decoder.InsertionDecoder.Params()
    p.Define('insertion', insertion.SymbolInsertionLayer.Params(),
             'Insertion specifications (i.e., rollin and oracle policy).')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(InsertionModel, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      self.CreateChild('insertion', p.insertion)

  def _SampleCanvasAndTargets(self, x, x_paddings):
    """Sample a canvas and its corresponding targets.

    Args:
      x: A Tensor representing the canvas.
      x_paddings: A Tensor representing the canvas paddings.

    Returns:
      A `NestedMap` capturing the new sampled canvas and its targets.
    """
    p = self.params

    # TODO(williamchan): Consider grabbing `eos_id` from `x` instead of `p`.
    eos_id = p.decoder.target_eos_id

    # Sample a canvas (and it's corresponding targets).
    return self.insertion.FProp(None, x, x_paddings, eos_id, True)

  def _CreateCanvasAndTargets(self, batch):
    # pyformat: disable
    """Create the canvas and targets.

    Args:
      batch: A `.NestedMap`.

        - src: A `.NestedMap`.
          - ids: The source ids, ends in <eos>.
          - paddings: The source paddings.

        - tgt: A `.NestedMap`.
          - ids: The target ids, ends in <eos>.
          - paddings: The target paddings.

    Returns:
      A `NestedMap`.
        - canvas: The canvas (based off of the `rollin_policy`) of shape
          [batch_size, c_dim].
        - canvas_paddings: The paddings of `canvas_indices`.
        - target_indices: The target indices (i.e., use these indices to
          tf.gather_nd the log-probs). Optional, only during training.
        - target_weights: The target weights. Optional, only during training.
    """
    # pyformat: enable
    p = self.params

    if not self.do_eval:
      # Sample our src and tgt canvas.
      src_descriptor = self._SampleCanvasAndTargets(batch.src.ids,
                                                    batch.src.paddings)
      tgt_descriptor = self._SampleCanvasAndTargets(batch.tgt.ids,
                                                    batch.tgt.paddings)

      # Offset the src ids (to unshare embeddings between src/tgt). Note, we
      # only offset the canvas ids, but we do not offset the vocab ids. This
      # will result in unshared embeddings, but shared softmax. This is due to
      # GPU/TPU memory limitations, empirically it is known that unsharing
      # everything results in better performance.
      vocab_size = p.decoder.softmax.num_classes
      src_descriptor.canvas = tf.where(
          tf.equal(src_descriptor.canvas_paddings, 0),
          src_descriptor.canvas + vocab_size, src_descriptor.canvas)

      # Offset the tgt indices (need shift according to src length).
      batch_size = py_utils.GetShape(batch.src.ids)[0]
      # `target_batch` is a [num_targets, batch_size] tensor where each row
      # identifies which batch the target belongs to. Note the observation that,
      # tf.reduce_sum(target_batch, 1) == 1 \forall rows.
      target_batch = tf.cast(
          tf.equal(
              tf.expand_dims(tf.range(batch_size), 0),
              tf.expand_dims(tgt_descriptor.target_indices[:, 0], 1)), tf.int32)
      src_lens = tf.cast(
          tf.reduce_sum(1 - src_descriptor.canvas_paddings, 1), tf.int32)
      # `tgt_offset` is shape [num_targets] where each entry corresponds to the
      # offset needed for that target (due to the source length).
      tgt_offset = tf.matmul(target_batch, tf.expand_dims(src_lens, 1))
      # We shift the tgt slot without touching the batch or vocab.
      tgt_descriptor.target_indices += tf.concat(
          [tf.zeros_like(tgt_offset), tgt_offset,
           tf.zeros_like(tgt_offset)], 1)

      # The canvas is simply the sequence-level concat of the src and tgt.
      canvas, canvas_paddings = insertion.SequenceConcat(
          src_descriptor.canvas, src_descriptor.canvas_paddings,
          tgt_descriptor.canvas, tgt_descriptor.canvas_paddings)
      target_indices = tf.concat(
          [src_descriptor.target_indices, tgt_descriptor.target_indices], 0)
      target_weights = tf.concat(
          [src_descriptor.target_weights, tgt_descriptor.target_weights], 0)

      return py_utils.NestedMap(
          canvas=canvas,
          canvas_paddings=canvas_paddings,
          target_indices=target_indices,
          target_weights=target_weights)

  def ComputePredictions(self, theta, batch):
    # pyformat: disable
    """Compute the model predictions.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      batch: A `.NestedMap`.

        - src: A `.NestedMap`.
          - ids: The source ids, ends in <eos>.
          - paddings: The source paddings.

        - tgt: A `.NestedMap`.
          - ids: The target ids, ends in <eos>.
          - paddings: The target paddings.

    Returns:
      A `.NestedMap`.
        - outputs: The contextualized output vectors of shape
          [batch_size, time_dim, model_dim].
        - tgt: A `.NestedMap` (optional, only during training).
          - ids: The canvas ids.
          - paddings: The canvas paddings.
          - target_indices: The target indices.
          - target_weights: The target weights.
    """
    # pyformat: enable
    p = self.params

    # TODO(williamchan): Currently, we only support KERMIT mode (i.e., no
    # encoder, unified architecture).
    assert not p.encoder

    # Sometimes src and tgt have different types. We reconcile here and use
    # int32.
    batch.src.ids = tf.cast(batch.src.ids, tf.int32)
    batch.tgt.ids = tf.cast(batch.tgt.ids, tf.int32)

    canvas_and_targets = self._CreateCanvasAndTargets(batch)
    batch = py_utils.NestedMap(
        tgt=py_utils.NestedMap(
            ids=canvas_and_targets.canvas,
            paddings=canvas_and_targets.canvas_paddings))

    predictions = super(InsertionModel, self).ComputePredictions(theta, batch)

    if not self.do_eval:
      predictions.tgt = py_utils.NestedMap(
          ids=canvas_and_targets.canvas,
          paddings=canvas_and_targets.canvas_paddings,
          target_indices=canvas_and_targets.target_indices,
          target_weights=canvas_and_targets.target_weights)

    return predictions


class TransformerBatchMajorModel(MTBaseModel):
  """Transformer Model with batch major encoder and decoder.

  Implements Attention is All You Need:
  https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super(TransformerBatchMajorModel, cls).Params()
    p.encoder = encoder.TransformerBatchMajorEncoder.Params()
    p.decoder = decoder.TransformerBatchMajorDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Transformer batch-major model constructor.

    Args:
      params: Params used to construct this model.

    Raises:
      ValueError: If the decoder source_dim is different from the encoder
        model_dim.
    """
    super(TransformerBatchMajorModel, self).__init__(params)
    p = self.params
    if p.encoder.model_dim != p.decoder.source_dim:
      raise ValueError('The source_dim of Transformer decoder must be the '
                       'same as the model_dim of Transformer encoder.')
