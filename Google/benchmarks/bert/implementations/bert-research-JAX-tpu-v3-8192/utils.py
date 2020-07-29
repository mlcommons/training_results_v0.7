# Lint as: python3
"""Utility functions for JAX modeling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from flax import nn
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf
from REDACTED.tensorflow_models.mlperf.models.rough.bert import modeling
from REDACTED.tensorflow_models.official.nlp.bert import bert_models
from REDACTED.tensorflow_models.official.nlp.bert import configs


def apply_activation(intermediate_output, intermediate_activation):
  """Applies selected activation function to intermediate output."""
  if intermediate_activation is None:
    return intermediate_output

  if intermediate_activation == 'gelu':
    intermediate_output = nn.gelu(intermediate_output)
  elif intermediate_activation == 'relu':
    intermediate_output = nn.relu(intermediate_output)
  elif intermediate_activation == 'sigmoid':
    intermediate_output = nn.sigmoid(intermediate_output)
  elif intermediate_activation == 'softmax':
    intermediate_output = nn.softmax(intermediate_output)
  elif intermediate_activation == 'celu':
    intermediate_output = nn.celu(intermediate_output)
  elif intermediate_activation == 'elu':
    intermediate_output = nn.elu(intermediate_output)
  elif intermediate_activation == 'log_sigmoid':
    intermediate_output = nn.log_sigmoid(intermediate_output)
  elif intermediate_activation == 'log_softmax':
    intermediate_output = nn.log_softmax(intermediate_output)
  elif intermediate_activation == 'soft_sign':
    intermediate_output = nn.soft_sign(intermediate_output)
  elif intermediate_activation == 'softplus':
    intermediate_output = nn.softplus(intermediate_output)
  elif intermediate_activation == 'swish':
    intermediate_output = nn.swish(intermediate_output)
  elif intermediate_activation == 'tanh':
    intermediate_output = jnp.tanh(intermediate_output)
  else:
    raise NotImplementedError('%s activation function is not yet supported.' %
                              intermediate_activation)

  return intermediate_output


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope('cls/predictions'):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope('transform'):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        'output_bias',
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope('cls/seq_relationship'):
    output_weights = tf.get_variable(
        'output_weights',
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        'output_bias', shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_tf_config(config_path):
  """Returns TF Bert config..

  Args:
    config_path: path to TF mlperf model configuration file
  Returns:
    tf_config: dictionary tf model configurations
  """
  return modeling.BertConfig.from_json_file(config_path).__dict__


def get_mlperf_model_variables(config_path, init_checkpoint):
  """Return tf mlperf model parameters in a dictionary format.

  Use get_tf_model_variables if using kerasBERT checkpoint. This function works
  for mlperf model in: //third_party/tensorflow_models/mlperf/models/rough/bert.

  Args:
    config_path: path to TF mlperf model configuration file
    init_checkpoint: path to saved TF mlperf model checkpoint

  Returns:
    tf_config: dictionary tf model configurations
    tf_variables: dictionary of tf variables
    tf_model: tensorflow BERT model generated using input config and checkpoint
  """
  # Load saved model configuration
  bert_config = modeling.BertConfig.from_json_file(config_path)
  seq_length = bert_config.max_position_embeddings
  tf_variables = {}
  max_predictions_per_seq = 76

  # Generate BERT TF model and initiate variable update from checkpoint
  graph = tf.Graph()
  sess = tf.Session(graph=graph)
  with graph.as_default():
    input_ids = tf.zeros((4, seq_length), dtype=tf.int32)
    input_mask = tf.zeros((4, seq_length), dtype=tf.int32)
    segment_ids = tf.zeros((4, seq_length), dtype=tf.int32)
    masked_lm_positions = tf.zeros((4, max_predictions_per_seq), dtype=tf.int32)
    masked_lm_ids = tf.zeros((4, max_predictions_per_seq), dtype=tf.int32)
    masked_lm_weights = tf.zeros((4, max_predictions_per_seq), dtype=tf.float32)
    next_sentence_labels = tf.zeros((4), dtype=tf.int32)
    tf_model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=True)
    (masked_lm_loss, _,
     _) = get_masked_lm_output(bert_config, tf_model.get_sequence_output(),
                               tf_model.get_embedding_table(),
                               masked_lm_positions, masked_lm_ids,
                               masked_lm_weights)

    (next_sentence_loss, _,
     _) = get_next_sentence_output(bert_config, tf_model.get_pooled_output(),
                                   next_sentence_labels)
    _ = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()
    (assignment_map,
     _) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
    sess.run(tf.initializers.global_variables())
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
      tf_variables[var.name[:-2]] = val

  tf_config = bert_config.__dict__

  return tf_config, tf_variables, tf_model


def get_tf_model_variables(config_path, init_checkpoint):
  """Return tf model parameters in a dictionary format.

  Args:
    config_path: path to TF model configuration file
    init_checkpoint: path to saved TF model checkpoint

  Returns:
    tf_config: dictionary tf model configurations
    tf_variables: dictionary of tf variables
    tf_model: tensorflow BERT model generated using input config and checkpoint
  """
  # Load saved model configuration
  config = configs.BertConfig.from_json_file(config_path)

  # Generate BERT TF model and initiate variable update from checkpoint
  seq_len = 20
  _, tf_model = bert_models.squad_model(config, seq_len)
  checkpoint = tf.train.Checkpoint(model=tf_model)
  checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

  tf_config = config.__dict__
  tf_variables = {v.name: v.numpy() for v in tf_model.variables}

  return tf_config, tf_variables, tf_model


def convert_tf_config_to_jax_bert(config):
  """Convert TF BERT model config to be compatible with JAX BERT model.

  Args:
    config: dictionary of TF model configurations

  Returns:
    dictionary of param names and values compatible with JAX BERT model
  """
  unnecessary_keys = ['initializer_range', 'backward_compatible',
                      'embedding_size']
  for key in unnecessary_keys:
    if key in config:
      config.pop(key)

  # change TF parameter names to match JAX parameter names
  mapping = {
      'attention_dropout_rate': 'attention_probs_dropout_prob',
      'hidden_activation': 'hidden_act',
      'dropout_rate': 'hidden_dropout_prob',
      'emb_dim': 'hidden_size',
      'mlp_dim': 'intermediate_size',
      'max_len': 'max_position_embeddings',
      'num_heads': 'num_attention_heads',
      'num_layers': 'num_hidden_layers'
  }
  for jax_key, tf_key in mapping.items():
    config[jax_key] = config.pop(tf_key)

  return config


def convert_mlperf_param_dict_to_jax(tf_params, emb_dim, num_heads):
  """Modify TF mlperf model parameter dict to be compatible with JAX parameter dict.

  Convert parameter names in tf_params to match JAX parameter names and create
  a nested dictionary of parameters for each layer in the model using `/` in
  each key as a delimeter.
  This function uses mlperf model naming convention. Use
  convert_tf_param_dict_to_jax when using kerasBERT model configuration.

  Args:
    tf_params: TF parameter dict, a flat dict with `/` in keys indicating nested
      layers in the model.
    emb_dim: number of embedding dims
    num_heads: number of attention heads in BERT model

  Returns:
    outer_dict: nested parameter dict with JAX compatible names and layers
  """
  jax_params = {}
  # mapping between mlperf model and JAX model
  # works for model in //third_party/tensorflow_models/mlperf/models/rough/bert
  tf_key_to_jax_key = [
      ('cls/seq_relationship/', 'classification/predictions_transform_logits/'),
      ('output_weights', 'kernel'),
      ('transform_logits/output_bias', 'transform_logits/bias'),
      ('cls/predictions/', 'masked_lm/cls_predictions_'),
      ('transform/dense', 'transform_dense'),
      ('transform/LayerNorm', 'transform_layernorm'),
      ('predictions_output_bias', 'predictions_output_bias/bias'),
      ('bert/embeddings/word_embeddings', 'word_embeddings/embedding'),
      ('bert/', 'transformer_encoder/'),
      ('embeddings/token_type_embeddings', 'type_embeddings/embedding'),
      ('embeddings/position_embeddings', 'position_embeddings/embedding'),
      ('attention/self', 'self_attention'),
      ('attention/output', 'self_attention_output'),
      ('layer_norm/layer_norm_', 'layer_norm/'),
      ('output/LayerNorm', 'output_layer_norm'),
      ('intermediate/dense', 'intermediate'),
      ('output/dense', 'output'),
      ('pooler/dense/', 'pooler_transform/'),
      ('self_attention_output_layer_norm', 'self_attention_layer_norm'),
      ('embeddings/LayerNorm', 'embeddings_layer_norm'),
      ('encoder/layer', 'encoder_layer'),
      (':0', ''),
      ('beta', 'bias'),
      ('gamma', 'scale')
  ]
  for tf_key, val in tf_params.items():
    jax_key = tf_key
    for tf_name, jax_name in tf_key_to_jax_key:
      jax_key = jax_key.replace(tf_name, jax_name)

    # Reshape kernels if necessary
    jax_params[jax_key] = tf_params[tf_key]
    if 'self_attention_output/kernel' in jax_key:
      param = tf_params[tf_key]
      jax_params[jax_key] = param.reshape(
          (num_heads, -1, emb_dim))

  # jax position embedding kernel has additional dimension
  pos_embedding = jax_params[
      'transformer_encoder/position_embeddings/embedding']
  jax_params[
      'transformer_encoder/position_embeddings/embedding'] = pos_embedding[
          np.newaxis, ...]

  # convert flat param dict into nested dict using `/` as delimeter
  outer_dict = {}
  for key, val in jax_params.items():
    tokens = key.split('/')
    inner_dict = outer_dict
    # each token except the very last should add a layer to the nested dict
    for token in tokens[:-1]:
      if token not in inner_dict:
        inner_dict[token] = {}
      inner_dict = inner_dict[token]
    inner_dict[tokens[-1]] = val

  return outer_dict


def convert_tf_param_dict_to_jax(tf_params):
  """Modify TF parameter dict to be compatible with JAX parameter dict.

  Convert parameter names in tf_params to match JAX parameter names and create
  a nested dictionary of parameters for each layer in the model using `/` in
  each key as a delimeter.

  Args:
    tf_params: TF parameter dict, a flat dict with `/` in keys indicating nested
      layers in the model.

  Returns:
    outer_dict: nested parameter dict with JAX compatible names and layers
  """
  jax_params = {}
  tf_key_to_jax_key = [
      ('embeddings/layer_norm', 'embeddings_layer_norm'),
      ('transformer/layer', 'encoder_layer'), ('embeddings:0', 'embedding'),
      (':0', ''), ('beta', 'bias'), ('gamma', 'scale'),
      ('position_embedding/', 'position_embeddings/')
  ]
  for tf_key in tf_params:
    jax_key = tf_key
    for tf_name, jax_name in tf_key_to_jax_key:
      jax_key = jax_key.replace(tf_name, jax_name)

    jax_params[jax_key] = tf_params[tf_key]

  # jax position embedding kernel has additional dimension
  pos_embedding = jax_params['position_embeddings/embedding']
  jax_params['position_embeddings/embedding'] = pos_embedding[np.newaxis, ...]

  # convert flat param dict into nested dict using `/` as delimeter
  outer_dict = {}
  for key, val in jax_params.items():
    tokens = key.split('/')
    inner_dict = outer_dict
    # each token except the very last should add a layer to the nested dict
    for token in tokens[:-1]:
      if token not in inner_dict:
        inner_dict[token] = {}
      inner_dict = inner_dict[token]
    inner_dict[tokens[-1]] = val

  # this layer doesn't have parameters, but key is required to be present
  outer_dict['self_attention_mask'] = 0.

  return outer_dict
