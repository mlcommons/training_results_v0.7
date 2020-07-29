# Lint as: python3
"""Masked LM implementation in JAX based on official TF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from typing import List
from flax import nn
import jax.numpy as jnp
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import utils
LAYER_NORM_EPSILON = 1e-12


class Bias(nn.Module):
  """Adds a bias term to an input."""

  def apply(self,
            inputs: List[float],
            dtype: jnp.dtype = jnp.float32,
            initializers: List[float] = nn.initializers.zeros):
    """Applies bias layer."""
    bias = self.param('bias', inputs.shape[1:], initializers)
    outputs = inputs.astype(dtype) + bias.astype(dtype)
    assert outputs.dtype == dtype
    return outputs


def GatherIndexes(
    sequence_tensor: List[List[float]],
    positions: List[int]):
  """Applies gather indexes layer.

  Args:
    sequence_tensor: Sequence output of `BertModel` layer of shape
      (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
      hidden units of `BertModel` layer.
    positions: Positions ids of tokens in sequence to mask for pretraining
      of with dimension (batch_size, num_predictions) where
      `num_predictions` is maximum number of tokens to mask out and predict
      per each sequence.

  Returns:
    Masked out sequence tensor of shape (batch_size * num_predictions,
    num_hidden).
  """
  batch_size, seq_length, width = sequence_tensor.shape
  flat_offsets = jnp.reshape(jnp.arange(batch_size) * seq_length, [-1, 1])
  flat_positions = jnp.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = jnp.reshape(sequence_tensor,
                                     [batch_size * seq_length, width])
  output_tensor = jnp.take(flat_sequence_tensor, flat_positions, axis=0)

  return output_tensor


class MaskedLM(nn.Module):
  """Masked language model network head for BERT modeling."""

  def apply(self,
            sequence_data: List[float],
            masked_lm_positions: List[int],
            input_width: int,
            num_predictions: int,
            embedding_table: List[float],
            activation=None,
            kernel_initializer: List[float] = nn.initializers.xavier_uniform(),
            dtype: jnp.dtype = jnp.float32,
            output='logits'):
    """Applies masked language model layer on transformer encoder output.

    Args:
      sequence_data: input to this layer, cls output of transformer encoder
      masked_lm_positions: input to this layer, masked positions
      input_width: innermost dimension of the input tensor to this network
      num_predictions: number of predictions to make per sequence.
      embedding_table: embedding table to use for the embedding layer
      activation: activation, if any, for the dense layer in this network
      kernel_initializer: initializer for dense layer kernel
      dtype: datatype for the activiations, jnp.bfloat16 or jnp.float32
      output: output type for the layer. Can be either 'logits' or 'predictions'

    Returns:
      logits or predictions based on the selected output type
    """
    _, hidden_size = embedding_table.shape
    masked_lm_input = GatherIndexes(sequence_data, masked_lm_positions)

    lm_data = nn.Dense(
        masked_lm_input,
        hidden_size,
        kernel_init=kernel_initializer,
        dtype=dtype,
        name='cls_predictions_transform_dense')
    assert lm_data.dtype == dtype

    if activation:
      lm_data = utils.apply_activation(lm_data, activation)
    assert lm_data.dtype == dtype

    lm_data = nn.LayerNorm(
        lm_data,
        epsilon=LAYER_NORM_EPSILON,
        dtype=dtype,
        name='cls_predictions_transform_layernorm')
    assert lm_data.dtype == dtype

    lm_data = jnp.matmul(lm_data, jnp.transpose(embedding_table).astype(dtype))
    assert lm_data.dtype == dtype

    logits = Bias(lm_data, name='cls_predictions_output_bias', dtype=dtype)
    assert logits.dtype == dtype

    if output == 'logits':
      return logits
    else:
      # Apply softmax on f32 data.
      predictions = utils.apply_activation(logits.astype(jnp.float32),
                                           'log_softmax')
      return predictions
