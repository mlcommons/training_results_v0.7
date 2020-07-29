# Lint as: python3
"""Transformer encoder implementation in JAX based on official TF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from typing import List
from absl import flags
from flax import nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import transformer_block

flags.DEFINE_bool(
    'use_one_hot_encodings', False, help='One hot encodings for embeddings.')
FLAGS = flags.FLAGS

LAYER_NORM_EPSILON = 1e-12


class KernelLayer(nn.Module):
  """Creates kernel parameter with provided dims."""

  def apply(self, kernel_dims, kernel_init):
    return self.param('kernel', kernel_dims, kernel_init)


class DenseParams(nn.Module):
  """Get parameters for dense layer.

  Bias parameters are merged to make them faster during allreduce
  communications. Kernels are still separated.
  A kernel parameter for layer i is at
    DenseParams.params['layer_%d'%i]['kernel'],
  And bias parameter for layer i is at
    DenseParams.params['bias'][i].
  """

  def apply(self, num_kernels, kernel_dims, bias_dims,
            kernel_init, bias_init):
    # Create an KernelList object to make it to be treated as a leaf in
    # jax.tree_map

    class KernelList:

      def __init__(self, kernels):
        self.kernels = kernels

      def __getitem__(self, idx):
        return self.kernels[idx]

    # Do not merge kernels.
    kernels = [KernelLayer(kernel_dims, kernel_init, name='layer_%d' % i)
               for i in range(num_kernels)]
    bias = self.param('bias', bias_dims, bias_init)
    return {'kernel': KernelList(kernels), 'bias': bias}


class SelfAttentionParams(nn.Module):
  """Get parameters for self attention.

  Biases are merged, kernels are separate for each layer.
  A kernel parameter for layername in ['key', 'query', 'value'], transformer
  layer i is at
    SelfAttentionParams.params[layername]['layer_%d'%i]['kernel'],
  And bias parameter for the same layer.
    SelfAttentionParams.params[layername]
  """

  def apply(self, num_kernels, kernel_dims, bias_dims, kernel_init, bias_init):
    parameters = {}
    for layername in ['key', 'query', 'value']:
      parameters[layername] = DenseParams(num_kernels, kernel_dims, bias_dims,
                                          kernel_init, bias_init,
                                          name=layername)
    return parameters


class LayerNormParams(nn.Module):
  """Get parameters for dense layer.

  Both scale and bias parameters are merged for all layers.
  A scale or bias parameter for transformer layer i is at
    LayerNormParams.params['scale'][i], or
    LayerNormParams.params['bias'][i].
  """

  def apply(self,
            scale_dims,
            bias_dims,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros):
    scale = self.param('scale', scale_dims, scale_init)
    bias = self.param('bias', bias_dims, bias_init)
    return {'scale': scale, 'bias': bias}


class TransformerParameters(nn.Module):
  """Returns the parameters for all transformer layers.

  Given the number of layers, and other bert config parameters returns the
  parameters so that small variables are merged together (biases in dense
  layers, and scale and bias parameters in layer norms). Larger size parameters,
  e.g. kernel parameters are stored individually.

  The hierarchy of the paramerters are the same as TF1 bert to make it easy
  to load from checkpoint.
  """

  def apply(self,
            num_layers,
            qkv_dim: int,
            mlp_dim: int,
            num_attention_heads: int,
            kernel_init: List[float] = nn.initializers.xavier_uniform(),
            bias_init: List[float] = nn.initializers.zeros):

    size_per_head = qkv_dim // num_attention_heads
    parameters = {}
    kernel_dims = (qkv_dim, num_attention_heads, size_per_head)
    bias_dims = (num_layers, num_attention_heads, size_per_head)
    parameters['self_attention'] = SelfAttentionParams(
        num_layers,
        kernel_dims,
        bias_dims,
        kernel_init,
        bias_init,
        name='self_attention')

    kernel_dims = (num_attention_heads, size_per_head, qkv_dim)
    bias_dims = (num_layers, qkv_dim,)
    parameters['self_attention_output'] = DenseParams(
        num_layers,
        kernel_dims, bias_dims, kernel_init, bias_init,
        name='self_attention_output')

    bias_dims = (num_layers, qkv_dim,)
    scale_dims = (num_layers, qkv_dim,)
    parameters['self_attention_layer_norm'] = LayerNormParams(
        scale_dims, bias_dims, name='self_attention_layer_norm')

    kernel_dims = (qkv_dim, mlp_dim)
    bias_dims = (num_layers, mlp_dim,)
    parameters['intermediate'] = DenseParams(
        num_layers,
        kernel_dims, bias_dims, kernel_init, bias_init,
        name='intermediate')

    kernel_dims = (mlp_dim, qkv_dim)
    bias_dims = (num_layers, qkv_dim,)
    parameters['output'] = DenseParams(
        num_layers,
        kernel_dims, bias_dims, kernel_init, bias_init,
        name='output')

    bias_dims = (num_layers, qkv_dim,)
    scale_dims = (num_layers, qkv_dim,)
    parameters['output_layer_norm'] = LayerNormParams(
        scale_dims, bias_dims, name='output_layer_norm')

    encoder_params = {}
    def slice_fn(i):
      return lambda x: x[i]
    for i in range(num_layers):
      encoder_params['encoder_layer_%d' % i] = jax.tree_map(slice_fn(i),
                                                            parameters)
    return encoder_params


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            dtype: jnp.dtype = jnp.float32,
            emb_init=nn.initializers.normal(stddev=1.0),
            use_one_hot_encodings=False):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      dtype: dtype to use for activations.
      emb_init: embedding initializer
      use_one_hot_encodings: whether to use one hot encodings to get embeddings.
    Returns:
      output which is embedded input data
    """
    use_one_hot_encodings = FLAGS.use_one_hot_encodings
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    embedding = embedding.astype(dtype)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      if use_one_hot_encodings:
        flat_input_ids = jnp.reshape(inputs, [-1])
        one_hot_input_ids = common_utils.onehot(
            flat_input_ids, num_classes=num_embeddings)
        embed_tab = jnp.matmul(one_hot_input_ids, embedding)
        embed_tab = jnp.reshape(embed_tab,
                                [inputs.shape[0], -1, embed_tab.shape[-1]])
        return embed_tab
      else:
        return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)

  @nn.base.module_method
  def get_embedding_table(self, **unused_kwargs):
    embedding_table = self.get_param('embedding')
    return embedding_table


def self_attention_mask(data, mask):
  """Creates 3D attention mask from 2D mask based on input data shape.

  Args:
    data: input data of shape `(batch_size, from_seq_length, ...)`
    mask: input mask of shape `(batch_size, to_seq_length)`

  Returns:
    attention_mask: `(batch_size, from_seq_length, to_seq_length)`
  """
  batch_size = data.shape[0]
  from_seq_length = data.shape[1]
  to_seq_length = mask.shape[1]

  attention_mask = jnp.broadcast_to(
      mask[:, jnp.newaxis, ...], (batch_size, from_seq_length, to_seq_length))

  return attention_mask


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs."""

  def apply(self,
            inputs: List[float],
            max_len: int = 2048,
            posemb_init: List[float] = nn.initializers.xavier_normal()):
    """Applies AddPositionEmbs module.

    Args:
      inputs: input data
      max_len: maximum possible length for the input
      posemb_init: positional embedding initializer

    Returns:
      output: `(1, min(length, max_len), inputs.shape[-1])`
    """
    assert inputs.ndim == 3, ('Number of dimention should be 3, but it is: %d' %
                              inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    pos_embedding = self.param('embedding', pos_emb_shape, posemb_init)
    return pos_embedding[:, :length, :]


class TransformerEncoder(nn.Module):
  """Transformer encoder implementation based on official TF model."""

  def apply(self,
            inputs: List[List[float]],
            vocab_size: int,
            type_vocab_size: int = 16,
            emb_dim: int = 768,
            mlp_dim: int = 3072,
            max_len: int = 512,
            num_heads: int = 12,
            num_layers: int = 12,
            train: bool = False,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            embedding_table: List[float] = None,
            hidden_activation: str = 'gelu',
            dtype: jnp.dtype = jnp.float32,
            kernel_initializer: List[float] = nn.initializers.xavier_uniform()):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data = [word_ids, mask, type_ids]
      vocab_size: int size of the token vocabulary
      type_vocab_size: int number of types that the 'type_ids' input can take
      emb_dim: int dimension of th embedding layers
      mlp_dim: int dimension of the mlp on top of attention block
      max_len: int maximum sequence length that this encoder can consume.
      num_heads: number of heads
      num_layers: number of transformer block layers
      train: boolean whether the model is being trained
      dropout_rate: float dropout rate
      attention_dropout_rate: float dropout rate for attention weights
      embedding_table: a shared embedding layer to use
      hidden_activation: activation function applied to intermediate layer
      dtype: the dtype of the computation (default: float32)
      kernel_initializer: initializer for dense layer kernels

    Returns:
      cls_output: pooled output of the encoder
      data: output from the last layer of transformer block
    """
    # Unpack inputs
    word_ids, mask, type_ids = inputs

    assert word_ids.ndim == 2  # (batch, len)
    word_ids = word_ids.astype('int32')
    type_ids = type_ids.astype('int32')

    # Embedding layers
    if embedding_table is None:
      embedding_table = Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          dtype=dtype,
          emb_init=kernel_initializer,
          name='word_embeddings')
    word_embeddings = embedding_table(word_ids)

    position_embeddings = AddPositionEmbs(
        word_embeddings,
        max_len=max_len,
        posemb_init=kernel_initializer,
        name='position_embeddings')

    type_embeddings = Embed(
        type_ids,
        num_embeddings=type_vocab_size,
        features=emb_dim,
        dtype=dtype,
        emb_init=kernel_initializer,
        name='type_embeddings')

    embeddings = word_embeddings + type_embeddings
    embeddings = embeddings + position_embeddings
    embeddings = nn.LayerNorm(
        embeddings, epsilon=LAYER_NORM_EPSILON, name='embeddings_layer_norm')
    embeddings = nn.dropout(
        embeddings,
        rate=dropout_rate,
        deterministic=not train)
    data = embeddings.astype(dtype)
    mask = mask.astype(dtype)
    # Transformer block
    attention_mask = self_attention_mask(data, mask).astype('bool')

    # Create parameter hierarchy as close as possible to tf1 bert,
    # to make it easier to load.
    encoder_params = TransformerParameters(
        num_layers,
        qkv_dim=emb_dim,
        mlp_dim=mlp_dim,
        num_attention_heads=num_heads,
        kernel_init=kernel_initializer,
        name='encoder_layer_common')

    for i in range(num_layers):
      data = transformer_block.transformer_block(
          data,
          encoder_params['encoder_layer_%d'%i],
          qkv_dim=emb_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          padding_mask=attention_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          intermediate_activation=hidden_activation,
          kernel_initializer=kernel_initializer,
          dtype=dtype,
          deterministic=not train)
    assert data.dtype == dtype

    first_token_tensor = jnp.squeeze(data[:, 0:1, :], axis=1)
    assert first_token_tensor.dtype == dtype
    cls_output = nn.Dense(
        first_token_tensor,
        emb_dim,
        kernel_init=kernel_initializer,
        dtype=dtype,
        name='pooler_transform')
    assert cls_output.dtype == dtype
    cls_output = jnp.tanh(cls_output)
    assert cls_output.dtype == dtype
    return data, cls_output
