# Lint as: python3
"""Transformer block implementation in JAX based on official TF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

import math
from typing import List
from typing import TypeVar
from absl import flags
from flax import nn
from flax.nn import attention
import jax
from jax import lax
import jax.numpy as jnp
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import utils

flags.DEFINE_bool(
    'use_einsum', False, help='Whether to use einsums instead dense generals.')
FLAGS = flags.FLAGS

Precision = TypeVar('Precision', bound=lax.Precision)
NEG_INFINITY = -10000.0
LAYER_NORM_EPSILON = 1e-12


class Dense3D(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self, input_tensor, num_attention_heads, size_per_head,
            kernel_init, bias_init, dtype):
    """A dense layer with 3D kernel.

    Args:
      input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
      num_attention_heads: Number of attention heads.
      size_per_head: The size per attention head.
      kernel_init: Kernel initializer.
      bias_init: Bias initializer.
      dtype: data type.
    Returns:
      float logits Tensor.
    """
    last_dim = input_tensor.shape[-1]
    w = self.param('kernel', (last_dim, num_attention_heads * size_per_head),
                   kernel_init)

    w = jnp.reshape(w, [last_dim, num_attention_heads, size_per_head])
    w = w.astype(dtype)
    b = self.param('bias', (num_attention_heads * size_per_head,),
                   bias_init)

    b = jnp.reshape(b, [num_attention_heads, size_per_head])
    b = b.astype(dtype)
    ret = jnp.einsum('abc,cde->abde', input_tensor, w)
    ret += b
    return ret


class Dense2D(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self, input_tensor, output_size, kernel_init, bias_init, dtype):
    """A dense layer with 3D kernel.

    Args:
      input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
      output_size: Number of attention heads.
      kernel_init: Kernel initializer.
      bias_init: Bias initializer.
      dtype: data type.
    Returns:
      float logits Tensor.
    """
    last_dim = input_tensor.shape[-1]
    w = self.param('kernel', (last_dim, output_size), kernel_init)
    w = w.astype(dtype)
    b = self.param('bias', (output_size,), bias_init)
    b = b.astype(dtype)
    ret = jnp.einsum('abc,cd->abd', input_tensor, w)
    ret += b
    return ret


def self_attention(inputs,
                   variable_dictionary,
                   num_heads: int,
                   qkv_features: int = None,
                   padding_mask: List[bool] = None,
                   dropout_rate: float = 0.,
                   deterministic: bool = False,
                   precision: Precision = None,
                   kernel_init: List[float] = nn.linear.default_kernel_init,
                   bias_init: List[float] = nn.initializers.zeros,
                   dtype: jnp.dtype = jnp.float32,
                   bias: bool = True):
  """Applies Multi-head self-attention on the input data.

  Args:
    inputs: input data of shape `[bs, dim1, dim2, ..., dimN, features]`.
    variable_dictionary: Parameter dictionary.
    num_heads: number of attention heads. Features (i.e. inputs.shape[-1])
      should be divisible by the number of heads.
    qkv_features: dimension of the key, query, and value.
    padding_mask: boolean specifying tokens that are pad token.
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    dtype: datatype for the activiations, jnp.bfloat16 or jnp.float32
    bias: bool: whether pointwise QKVO dense transforms use bias.

  Returns:
    output of shape `[bs, dim1, dim2, ..., dimN, features//num_heads]`.
  """

  features = inputs.shape[-1]
  qkv_features = qkv_features or features

  assert qkv_features % num_heads == 0, (
      'Memory dimension must be divisible by number of heads.')
  head_dim = qkv_features // num_heads
  inputs = inputs.astype(dtype)
  if FLAGS.use_einsum:
    dense_module = Dense3D
  else:
    dense_module = attention.DenseGeneral

  query = dense_module.call(
      variable_dictionary['query'],
      inputs,
      axis=-1,
      features=(num_heads, head_dim),
      kernel_init=kernel_init,
      bias_init=bias_init,
      bias=bias,
      precision=precision,
      dtype=dtype,
      name='query')
  query = jnp.multiply(query, 1.0 / math.sqrt(float(head_dim)))
  key = dense_module.call(
      variable_dictionary['key'],
      inputs,
      axis=-1,
      features=(num_heads, head_dim),
      kernel_init=kernel_init,
      bias_init=bias_init,
      bias=bias,
      precision=precision,
      dtype=dtype,
      name='key')
  value = dense_module.call(
      variable_dictionary['value'],
      inputs,
      axis=-1,
      features=(num_heads, head_dim),
      kernel_init=kernel_init,
      bias_init=bias_init,
      bias=bias,
      precision=precision,
      dtype=dtype,
      name='value')

  assert query.dtype == dtype
  assert key.dtype == dtype
  assert value.dtype == dtype
  # get raw attention scores from dot product between key and query
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_heads`
  #   H = `head_dim` (qkv_features // num_heads)
  attention_scores = jnp.einsum('BTNH,BFNH->BNFT', key, query)
  assert attention_scores.dtype == dtype

  assert attention_scores.dtype == dtype
  # create attention masks
  if padding_mask is not None:
    assert padding_mask.dtype == bool, ('Mask should have bool type.')
    attention_mask = jnp.expand_dims(padding_mask, axis=1)
    adder = (1.0 - attention_mask) * NEG_INFINITY
    attention_scores += adder.astype(dtype)
  assert attention_scores.dtype == dtype

  attention_scores = attention_scores - lax.stop_gradient(
      jnp.max(attention_scores, axis=-1, keepdims=True))
  attention_scores = jnp.exp(attention_scores)
  attention_sum = jnp.sum(attention_scores, axis=-1, keepdims=True)

  keep_prob = 1 - dropout_rate
  if not deterministic:
    keep_mask = jax.random.bernoulli(nn.make_rng(), keep_prob,
                                     attention_scores.shape).astype(dtype)
    assert keep_mask.dtype == dtype
    attention_probs = jnp.multiply(keep_mask, attention_scores)
  else:
    attention_probs = attention_scores

  assert attention_probs.dtype == dtype

  attention_probs = jnp.einsum('BNFT,BTNH->BFNH', attention_probs, value)
  assert attention_probs.dtype == dtype
  attention_probs = attention_probs / jnp.transpose(attention_sum, [0, 2, 1, 3])

  # split mask and scaling ops in dropout
  # move the scaling from dropout to here to save same mul ops
  # TODO(yuemmawang) automate this optimization in xla
  if not deterministic:
    scale = 1 / keep_prob
    if dtype == jnp.bfloat16:
      scale = jnp.bfloat16(scale)
    attention_probs = jnp.multiply(attention_probs, scale)
  assert attention_probs.dtype == dtype

  return attention_probs


def transformer_block(
    inputs,
    variable_dictionary,
    qkv_dim: int,
    mlp_dim: int,
    num_heads: int,
    padding_mask: bool = None,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.1,
    intermediate_activation: str = 'gelu',
    kernel_initializer: List[float] = nn.initializers.xavier_uniform(),
    dtype: jnp.dtype = jnp.float32,
    deterministic: bool = False):
  """Applies TransformerBlock module.

  Args:
    inputs: input data
    variable_dictionary: Parameter dictionary.
    qkv_dim: int dimension of the query/key/value
    mlp_dim: int dimension of the mlp on top of attention block
    num_heads: number of heads
    padding_mask: bool, mask padding tokens
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    intermediate_activation: activation function applied to intermediate layer
    kernel_initializer: initializer for dense layer kernels
    dtype: datatype for the activiations, jnp.bfloat16 or jnp.float32
    deterministic: bool, deterministic or not (to apply dropout)

  Returns:
    output after transformer block.
  """

  # Attention block.
  assert inputs.ndim == 3
  attention_output = self_attention(
      inputs,
      variable_dictionary['self_attention'],
      num_heads=num_heads,
      qkv_features=qkv_dim,
      padding_mask=padding_mask,
      kernel_init=kernel_initializer,
      dropout_rate=attention_dropout_rate,
      deterministic=deterministic,
      dtype=dtype)
  attention_output = nn.linear.DenseGeneral.call(
      variable_dictionary['self_attention_output'],
      attention_output,
      features=qkv_dim,
      axis=(-2, -1),
      kernel_init=kernel_initializer,
      dtype=dtype,
      name='self_attention_output')
  assert attention_output.dtype == dtype

  attention_output = nn.dropout(
      attention_output, rate=dropout_rate, deterministic=deterministic)
  assert attention_output.dtype == dtype
  attention_output = inputs + attention_output
  assert attention_output.dtype == dtype

  # Mlp block
  attention_output = nn.LayerNorm.call(
      variable_dictionary['self_attention_layer_norm'],
      attention_output,
      epsilon=LAYER_NORM_EPSILON,
      dtype=dtype,
      name='self_attention_layer_norm')
  assert attention_output.dtype == dtype
  if FLAGS.use_einsum:
    intermediate_output = Dense2D.call(
        variable_dictionary['intermediate'],
        attention_output,
        mlp_dim,
        kernel_initializer,
        bias_init=nn.initializers.zeros,
        dtype=dtype,
        name='intermediate')
  else:
    intermediate_output = nn.Dense.call(
        variable_dictionary['intermediate'],
        attention_output,
        mlp_dim,
        kernel_init=kernel_initializer,
        dtype=dtype,
        name='intermediate')

  assert intermediate_output.dtype == dtype
  intermediate_output = utils.apply_activation(intermediate_output,
                                               intermediate_activation)
  assert intermediate_output.dtype == dtype
  if FLAGS.use_einsum:
    layer_output = Dense2D.call(
        variable_dictionary['output'],
        intermediate_output, qkv_dim, kernel_initializer,
        bias_init=nn.initializers.zeros, dtype=dtype,
        name='output')
  else:
    layer_output = nn.Dense.call(
        variable_dictionary['output'],
        intermediate_output,
        qkv_dim,
        kernel_init=kernel_initializer,
        dtype=dtype,
        name='output')

  assert layer_output.dtype == dtype

  layer_output = nn.dropout(
      layer_output, rate=dropout_rate, deterministic=deterministic)
  assert layer_output.dtype == dtype

  layer_output = nn.LayerNorm.call(
      variable_dictionary['output_layer_norm'],
      layer_output + attention_output,
      epsilon=LAYER_NORM_EPSILON,
      name='output_layer_norm',
      dtype=dtype)
  assert layer_output.dtype == dtype
  return layer_output
