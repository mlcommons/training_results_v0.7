# Lint as: python3
"""Classification layer implementation in JAX based on official TF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from typing import List
from flax import nn
import jax.numpy as jnp
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import utils


default_kernel_init = nn.initializers.lecun_normal()


class TransposeDense(nn.Module):
  """A linear transformation applied over the last dimension of the input."""

  def apply(self,
            inputs,
            features,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=nn.initializers.zeros):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      features: the number of output features.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: bfloat16)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.

    Returns:
      The transformed input.
    """
    inputs = inputs.astype(dtype)
    kernel = self.param('kernel', (features, inputs.shape[-1]), kernel_init)
    kernel = kernel.astype(dtype)
    y = jnp.matmul(inputs, jnp.transpose(kernel, (1, 0)))
    if bias:
      bias = self.param('bias', (features,), bias_init)
      y = y + bias.astype(dtype)
    assert y.dtype == dtype
    return y


class Classification(nn.Module):
  """Classification network head for BERT modeling."""

  def apply(self,
            cls_output: List[float],
            input_width,
            num_classes,
            kernel_initializer: List[float] = nn.initializers.xavier_uniform(),
            transpose_weights=False,
            dtype: jnp.dtype = jnp.float32,
            output='logits'):
    """Applies classification on transformer encoder output.

    Args:
      cls_output: input to this layer, cls output of transformer encoder
      input_width: innermost dimension of the input tensor to this network
      num_classes: number of classes that this network should classify to.
      kernel_initializer: initializer for dense layer kernel
      transpose_weights: whether or not to transpose weights in Dense layer
      dtype: datatype for the activiations, jnp.bfloat16 or jnp.float32
      output: output type for the layer. Can be either 'logits' or 'predictions'

    Returns:
      logits or predictions based on the selected output type
    """
    if transpose_weights:
      logits = TransposeDense(
          cls_output,
          num_classes,
          kernel_init=kernel_initializer,
          dtype=dtype,
          name='predictions_transform_logits')
    else:
      logits = nn.Dense(
          cls_output,
          num_classes,
          kernel_init=kernel_initializer,
          dtype=dtype,
          name='predictions_transform_logits')
    assert logits.dtype == dtype
    if output == 'logits':
      return logits
    else:
      predictions = utils.apply_activation(logits.astype(jnp.float32),
                                           'log_softmax')
      return predictions
