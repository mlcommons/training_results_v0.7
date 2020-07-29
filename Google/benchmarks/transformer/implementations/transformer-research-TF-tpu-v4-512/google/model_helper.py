# Lint as: python2, python3
"""Some utilities for configuring models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from REDACTED.transformer_lingvo.lingvo import compat as tf
from REDACTED.transformer_lingvo.lingvo.core import base_input_generator
from REDACTED.transformer_lingvo.lingvo.core import base_model_params
from REDACTED.transformer_lingvo.lingvo.core import beam_search_helper
from REDACTED.transformer_lingvo.lingvo.core import hyperparams
from REDACTED.transformer_lingvo.lingvo.core import layers
from REDACTED.transformer_lingvo.lingvo.core import model_helper as lingvo_model_helper
from REDACTED.transformer_lingvo.lingvo.core import program as program_lib
from REDACTED.transformer_lingvo.lingvo.core import rnn_cell
import six
from six.moves import range

from REDACTED.transformer_lingvo.google import beam_search_tpu_helper


def ChangeToSimpleEmbedding(emb_params,
                            use_matmul=False,
                            use_3d_weight_tensor=False,
                            scale_sqrt_depth=False):
  """Returns a SimpleEmbeddingLayer param given emb_params.

  Args:
    emb_params: emb_params must be layers.EmbeddingLayer param.
    use_matmul: If True, uses a matmul between one hot vectors and the embedding
      matrix. Otherwise, reads one row per id.
    use_3d_weight_tensor: If true, uses a 3-d weight tensor by reshaping
      [num_rows, dim] -> [num_rows, dim // 128, 128].
    scale_sqrt_depth: If True, output is scaled with sqrt(embedding_dim).

  Returns:
    A SimpleEmbeddingLayer param. Note that use_matmul=True tends
    to use more memory.
  """
  assert emb_params.cls is layers.EmbeddingLayer or layers.SimpleEmbeddingLayer
  p = layers.SimpleEmbeddingLayer.Params()
  p.use_matmul = use_matmul
  p.use_3d_weight_tensor = use_3d_weight_tensor
  p.vocab_size = emb_params.vocab_size
  p.embedding_dim = emb_params.embedding_dim
  p.vn = emb_params.vn.Copy()
  p.params_init = emb_params.params_init.Copy()
  p.scale_sqrt_depth = scale_sqrt_depth
  return p


def ChangeToSimpleSoftmax(softmax_params):
  """Returns SimpleFullSoftmax param given softmax_params.

  Args:
    softmax_params: softmax_params must be layers.SoftmaxLayer's subclass'
      param.

  Returns:
    A SimpleFullSoftmax param.
  """
  p = layers.SimpleFullSoftmax.Params()
  p.vn = softmax_params.vn.Copy()
  p.params_init = softmax_params.params_init.Copy()
  p.input_dim = softmax_params.input_dim
  p.num_classes = softmax_params.num_classes
  p.num_shards = 1
  p.chunk_size = softmax_params.chunk_size
  return p



def ChangeToBeamSearchTpuHelper(beam_search_helper_params):
  """Returns BeamSearchTpuHelper params given beam_search_helper_params.

  Args:
    beam_search_helper_params: This must be
      beam_search_helper.BeamSearchHelper.Params().

  Returns:
    BeamSearchTpuHelper params.
  Raises:
    ValueError: If beam_search_helper_params.cls is not
        beam_search_helper.BeamSearchHelper.
  """
  if issubclass(beam_search_helper_params.cls,
                beam_search_tpu_helper.BeamSearchTpuHelper):
    return beam_search_helper_params
  if beam_search_helper_params.cls is not beam_search_helper.BeamSearchHelper:
    raise ValueError('Expected beam_search_helper_params.cls to be '
                     'beam_search_helper.BeamSearchHelper: %s' %
                     beam_search_helper_params.cls)
  p = beam_search_tpu_helper.BeamSearchTpuHelper.Params()
  p.num_hyps_per_beam = beam_search_helper_params.num_hyps_per_beam
  p.target_seq_length_ratio = beam_search_helper_params.target_seq_length_ratio
  p.length_normalization = beam_search_helper_params.length_normalization
  p.coverage_penalty = beam_search_helper_params.coverage_penalty
  p.valid_eos_max_logit_delta = (
      beam_search_helper_params.valid_eos_max_logit_delta)
  p.local_eos_threshold = beam_search_helper_params.local_eos_threshold
  p.beam_size = beam_search_helper_params.beam_size
  p.target_sos_id = beam_search_helper_params.target_sos_id
  p.target_eos_id = beam_search_helper_params.target_eos_id
  p.target_eoc_id = beam_search_helper_params.target_eoc_id
  p.merge_paths = beam_search_helper_params.merge_paths
  p.target_seq_len = beam_search_helper_params.target_seq_len
  p.batch_major_state = beam_search_helper_params.batch_major_state
  p.batch_major_compute = beam_search_helper_params.batch_major_compute
  p.short_seq_limit = beam_search_helper_params.short_seq_limit
  tf.logging.info('Running BeamSearchTpuHelper with params: %s ', p.ToText())
  return p


def FixateInputShape(input_params,
                     batch_size,
                     target_max_seqlen=None,
                     source_max_seqlen=None):
  """Adjusts the input_params so that it always produces fixed shaped output.

  Sets p.target_max_length to target_max_seqlen and p.source_max_length to
  source_max_seqlen if set. Only keep items in bucket_upper_bound that is <=
  max_seqlen where max_seqlen is source_max_seqlen if given; otherwise,
  target_max_seqlen.

  Args:
    input_params: The input params.
    batch_size: The input generator should always output the batch size.
    target_max_seqlen: Every batch should produce samples with this sequence
      length, and samples are padded to this length. Use
      input_params.bucket_upper_bound[-1] if not set.
    source_max_seqlen: Same effect as target_max_seqlen but for source if set.

  Returns:
    input_params itself.
  """
  p = input_params
  # Pad to fixed length, since otherwise the infeed queue can't be set up,
  # because the shape won't be known statically.
  #
  # Limit memory by throwing away large sequences.
  if not target_max_seqlen:
    assert p.bucket_upper_bound
    target_max_seqlen = p.bucket_upper_bound[-1]
  if source_max_seqlen:
    p.bucket_upper_bound = [
        x for x in p.bucket_upper_bound if x <= source_max_seqlen
    ]
    if not p.bucket_upper_bound:
      p.bucket_upper_bound = [source_max_seqlen]
  else:
    p.bucket_upper_bound = [
        x for x in p.bucket_upper_bound if x <= target_max_seqlen
    ]
    if not p.bucket_upper_bound:
      p.bucket_upper_bound = [target_max_seqlen]
  p.bucket_batch_limit = [batch_size] * len(p.bucket_upper_bound)
  p.pad_to_max_seq_length = True
  if source_max_seqlen:
    p.source_max_length = source_max_seqlen
  p.target_max_length = target_max_seqlen
  if hasattr(p, 'pad_and_set_target_shape'):
    p.pad_and_set_target_shape = True

  # Because every batch is padded to the max sequence length and has the same
  # batch size, they are shardable.
  p.remote.shardable_batch = True
  return p
