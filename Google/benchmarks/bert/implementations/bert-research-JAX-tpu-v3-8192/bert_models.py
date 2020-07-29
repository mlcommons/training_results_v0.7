# Lint as: python3
"""BERT model implementation in JAX based on official TF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from typing import List
from flax import nn
import jax.numpy as jnp
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import classification
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import masked_lm
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import transformer_encoder
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import utils


class PretrainModel(nn.Module):
  """Bert Model for pre-training."""

  def apply(self,
            inputs: List[List[float]],
            num_token_predictions: int,
            num_classes: int,
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
            hidden_activation: str = None,
            output: str = 'logits',
            dtype: jnp.dtype = jnp.float32,
            kernel_initializer: List[float] = nn.initializers.xavier_uniform()):
    """Applies pretraining on the inputs.

    Args:
      inputs: input data = [word_ids, mask, type_ids, lm_mask]
      num_token_predictions: int size of number of token predictions
      num_classes: int number of classes for classification
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
      hidden_activation: activation function applied to intermediate layer
      output: output type for the model. Can be either 'logits' or 'predictions'
      dtype: the dtype of the computation (default: float32)
      kernel_initializer: initializer for dense layer kernels

    Returns:
      (start_logits, end_logits): output from the squad model
    """
    masked_positions = inputs[-1][..., :num_token_predictions].astype('int32')

    embedding_table = transformer_encoder.Embed.shared(
        num_embeddings=vocab_size,
        features=emb_dim,
        dtype=dtype,
        emb_init=kernel_initializer,
        name='word_embeddings')
    sequence_output, cls_output = transformer_encoder.TransformerEncoder(
        inputs[:3],
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        emb_dim=emb_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        embedding_table=embedding_table,
        hidden_activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        dtype=dtype,
        name='transformer_encoder')

    sequence_output_length = sequence_output.shape[1]
    if sequence_output_length < num_token_predictions:
      raise ValueError(
          "The passed network's output length is %s, which is less than the "
          'requested num_token_predictions %s.' %
          (sequence_output_length, num_token_predictions))

    # Convert tensors to f32 here, as in tf model.
    sequence_output = sequence_output.astype(jnp.float32)
    cls_output = cls_output.astype(jnp.float32)

    lm_outputs = masked_lm.MaskedLM(
        sequence_output,
        masked_positions,
        input_width=sequence_output.shape[-1],
        num_predictions=num_token_predictions,
        embedding_table=embedding_table.get_embedding_table(),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        output=output,
        dtype=jnp.float32,
        name='masked_lm')
    sentence_outputs = classification.Classification(
        cls_output,
        input_width=cls_output.shape[-1],
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        transpose_weights=True,
        output=output,
        dtype=jnp.float32,
        name='classification')
    return lm_outputs, sentence_outputs


class SpanLabeling(nn.Module):
  """Span labeling layer for BERT modeling."""

  def apply(self,
            sequence_data: List[float],
            input_width: int,
            activation: str = None,
            kernel_initializer: List[float] = nn.initializers.xavier_uniform(),
            output: str = 'logits'):
    """Applies span labeling on transformer encoder output.

    Args:
      sequence_data: input to this layer, sequence output of transformer encoder
      input_width: last dimension of input tensor sequence_data
      activation: activation function applied on dense layer output
      kernel_initializer: initializer for dense layer kernel
      output: output type for the layer. Can be either 'logits' or 'predictions'

    Returns:
      start and end logits if output type is selected as 'logits'. Start and end
      predictions if output type is 'predictions'.
    """

    # This layer predicts start location and end location.
    intermediate_logits = nn.Dense(
        sequence_data,
        2,
        kernel_init=kernel_initializer,
        name='predictions_transform_logits')
    if activation:
      intermediate_logits = utils.apply_activation(intermediate_logits,
                                                   activation)

    start_logits, end_logits = jnp.transpose(intermediate_logits, [2, 0, 1])

    if output == 'logits':
      return start_logits, end_logits
    elif output == 'predictions':
      start_predictions = utils.apply_activation(start_logits, 'log_softmax')
      end_predictions = utils.apply_activation(end_logits, 'log_softmax')
      return start_predictions, end_predictions
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)


class SquadModel(nn.Module):
  """Squad Model implementation based on official TF model."""

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
            hidden_activation: str = 'gelu',
            output: str = 'logits',
            kernel_initializer: List[float] = nn.initializers.xavier_uniform()):
    """Applies Squad model on the inputs.

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
      hidden_activation: activation function applied to intermediate layer
      output: output type for the model. Can be either 'logits' or 'predictions'
      kernel_initializer: initializer for dense layer kernels

    Returns:
      (start_logits, end_logits): output from the squad model
    """

    sequence_output, _ = transformer_encoder.TransformerEncoder(
        inputs,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        emb_dim=emb_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        hidden_activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        name='transformer_encoder')

    start_logits, end_logits = SpanLabeling(
        sequence_output,
        input_width=sequence_output.shape[-1],
        kernel_initializer=kernel_initializer,
        output=output,
        name='span_labeling')

    return start_logits, end_logits
