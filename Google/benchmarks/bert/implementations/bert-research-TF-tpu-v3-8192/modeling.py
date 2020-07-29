"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding


def bfloat16_var_getter(getter, *args, **kwargs):
  """A custom getter function for bfloat16 variables.

  Variables maintain storage in float32.

  Args:
    getter: custom getter
    *args: arguments
    **kwargs: keyword arguments

  Returns:
    variables with the correct dtype.
  Raises:
    KeyError: if "dtype" is not provided as a kwarg.
  """
  requested_dtype = kwargs["dtype"]
  if requested_dtype == tf.bfloat16:
    kwargs["dtype"] = tf.float32
  var = getter(*args, **kwargs)
  # This if statement is needed to guard the cast, because batch norm
  # assigns directly to the return value of this custom getter. The cast
  # makes the return value not a variable so it cannot be assigned. Batch
  # norm variables are always in fp32 so this if statement is never
  # triggered for them.
  if var.dtype.base_dtype != requested_dtype:
    var = tf.cast(var, requested_dtype)
  return var


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               use_bfloat16_activation=False,
               num_partitions=1,
               scope="bert"):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      use_bfloat16_activation: (optional) bool. Whether to use bfloat16
              for activations.
      num_partitions: (optional) int32. Number of SPMD partitions.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert", reuse=tf.AUTO_REUSE):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.word_embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_bfloat16_activation=use_bfloat16_activation)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.word_embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        if use_bfloat16_activation:
          self.embedding_output = tf.cast(self.embedding_output, tf.bfloat16)

      with tf.variable_scope(
          "encoder",
          dtype=(tf.bfloat16 if use_bfloat16_activation else tf.float32)):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask, use_bfloat16_activation)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            num_partitions=num_partitions)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope(
          "pooler",
          dtype=(tf.bfloat16 if use_bfloat16_activation else tf.float32)):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_word_embedding_output(self):
    """Get output of the word(piece) embedding lookup.

    This is BEFORE positional embeddings and token type embeddings have been
    added.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the word(piece) embedding layer.
    """
    return self.word_embedding_output

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, rate=dropout_prob)
  return output


def layer_norm_vars(filters, layer_idx, total_layers):
  """Create Variables for layer norm."""
  if total_layers == 0:
    scale = tf.get_variable("gamma", filters, initializer=tf.ones_initializer())
    bias = tf.get_variable("beta", filters, initializer=tf.zeros_initializer())
  else:
    scale = tf.get_variable(
        "gamma", [total_layers, filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "beta", [total_layers, filters], initializer=tf.zeros_initializer())
    scale = tf.gather(scale, layer_idx)
    bias = tf.gather(bias, layer_idx)
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
      x, axes=[-1], keep_dims=True)
  mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss, None)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias


def layer_norm(x,
               layer_idx=0,
               total_layers=0,
               filters=None,
               epsilon=1e-12,
               name=None,
               reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(
      "LayerNorm" if not name else name,
      default_name="LayerNorm",
      values=[x],
      reuse=reuse):
    scale, bias = layer_norm_vars(filters, layer_idx, total_layers)
    return tf.cast(
        layer_norm_compute(tf.cast(x, tf.float32), epsilon, scale, bias),
        x.dtype)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, 0, 0, name=name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False,
                     use_bfloat16_activation=False,
                     num_partitions=1):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.
    use_bfloat16_activation: bool. If True, cast embedding to bfloat16.
    num_partitions: (optional) Number of SPMD partitions.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  if num_partitions > 1:
    embedding_table = xla_sharding.split(
        embedding_table, 0, num_partitions, use_sharding_op=True)

  if use_bfloat16_activation:
    embedding_table = tf.cast(embedding_table, tf.bfloat16)

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.cast(
        tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range)), output.dtype)
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(
        flat_token_type_ids,
        depth=token_type_vocab_size,
        dtype=input_tensor.dtype)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.cast(
          tf.get_variable(
              name=position_embedding_name,
              shape=[max_position_embeddings, width],
              initializer=create_initializer(initializer_range)), output.dtype)
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask,
                                          use_bfloat16_activation):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    use_bfloat16_activation: (optional) bool. Whether to use bfloat16
          activation.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
      tf.bfloat16 if use_bfloat16_activation else tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1],
      dtype=tf.bfloat16 if use_bfloat16_activation else tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def dense_layer_3d(input_tensor,
                   layer_idx,
                   total_layers,
                   num_attention_heads,
                   size_per_head,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 3D kernel.

  Args:
    input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
    layer_idx: the index of the current layer.
    total_layers: total number of layers.
    num_attention_heads: Number of attention heads.
    size_per_head: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """

  last_dim = get_shape_list(input_tensor)[-1]

  with tf.variable_scope(name, dtype=input_tensor.dtype):
    with tf.variable_scope("layer_%d" % layer_idx):
      w = tf.get_variable(
          name="kernel",
          shape=[last_dim, num_attention_heads, size_per_head],
          initializer=initializer)
    b = tf.get_variable(
        name="bias",
        shape=[total_layers, num_attention_heads, size_per_head],
        initializer=tf.zeros_initializer)
    b = tf.gather(b, layer_idx)

    ret = tf.einsum("abc,cde->abde", input_tensor, w)

    ret += b
    if activation is not None:
      return activation(ret)
    else:
      return ret


def dense_layer_3d_proj(input_tensor,
                        layer_idx,
                        total_layers,
                        hidden_size,
                        num_attention_heads,
                        head_size,
                        initializer,
                        activation,
                        name=None):
  """A dense layer with 3D kernel for projection.

  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    layer_idx: the index of the current layer.
    total_layers: total number of layers.
    hidden_size: The size of hidden layer.
    num_attention_heads: The size of output dimension.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  head_size = hidden_size // num_attention_heads
  with tf.variable_scope(name):
    with tf.variable_scope("layer_%d" % layer_idx):
      w = tf.get_variable(
          name="kernel",
          shape=[num_attention_heads, head_size, hidden_size],
          initializer=initializer)
    b = tf.get_variable(
        name="bias",
        shape=[total_layers, hidden_size],
        initializer=tf.zeros_initializer)
    b = tf.gather(b, layer_idx)

  ret = tf.einsum("BFNH,NHD->BFD", input_tensor, w)
  ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_2d(input_tensor,
                   layer_idx,
                   total_layers,
                   output_size,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 2D kernel.

  Args:
    input_tensor: Float tensor with rank 3.
    layer_idx: the index of the current layer.
    total_layers: total number of layers.
    output_size: The size of output dimension.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  last_dim = get_shape_list(input_tensor)[-1]
  with tf.variable_scope(name, dtype=input_tensor.dtype):
    with tf.variable_scope("layer_%d" % layer_idx):
      w = tf.get_variable(
          name="kernel", shape=[last_dim, output_size], initializer=initializer)
    b = tf.get_variable(
        name="bias",
        shape=[total_layers, output_size],
        initializer=tf.zeros_initializer)
    b = tf.gather(b, layer_idx)

  ret = tf.einsum("abc,cd->abd", input_tensor, w)
  ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def attention_layer(from_tensor,
                    to_tensor,
                    layer_idx,
                    total_layers,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    num_partitions=1):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with tf.einsum as follows:
    Input_tensor: [BFD]
    Wq, Wk, Wv: [DNH]
    Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
    K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
    V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
    attention_scores:[BNFT] = einsum('BFNH,BTNH>BNFT', Q, K) / sqrt(H)
    attention_probs:[BNFT] = softmax(attention_scores)
    context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
    Wout:[DNH]
    Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    layer_idx: the index of the current layer.
    total_layers: total number of layers.
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    num_partitions: (optional) Number of SPMD partitions.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  # `query_layer` = [B, F, N, H]
  query_layer = dense_layer_3d(from_tensor, layer_idx, total_layers,
                               num_attention_heads, size_per_head,
                               create_initializer(initializer_range), query_act,
                               name="query")

  # `key_layer` = [B, T, N, H]
  key_layer = dense_layer_3d(to_tensor, layer_idx, total_layers,
                             num_attention_heads, size_per_head,
                             create_initializer(initializer_range), key_act,
                             name="key")

  # `value_layer` = [B, T, N, H]
  value_layer = dense_layer_3d(to_tensor, layer_idx, total_layers,
                               num_attention_heads, size_per_head,
                               create_initializer(initializer_range), value_act,
                               name="value")
  if num_partitions > 1:
    # partition along the heads dimension
    query_layer = xla_sharding.split(
        query_layer, 2, num_partitions, use_sharding_op=True)
    key_layer = xla_sharding.split(
        key_layer, 2, num_partitions, use_sharding_op=True)
    value_layer = xla_sharding.split(
        value_layer, 2, num_partitions, use_sharding_op=True)

  query_layer = tf.multiply(query_layer, 1.0 / math.sqrt(float(size_per_head)))
  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_layer, query_layer)

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_scores = tf.cast(attention_scores, tf.float32)
  attention_scores = attention_scores - tf.stop_gradient(
      tf.reduce_max(attention_scores, -1, True))
  attention_scores = tf.exp(attention_scores)
  attention_sum = tf.reduce_sum(attention_scores, -1, True)
  attention_probs = tf.cast(attention_scores, key_layer.dtype)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  # Split mask and scaling ops in dropout
  random_u = tf.random_uniform(attention_probs.shape, dtype=tf.bfloat16)
  keep_mask = random_u >= attention_probs_dropout_prob
  keep_mask = tf.cast(keep_mask, dtype=attention_probs.dtype)

  attention_probs = tf.multiply(keep_mask, attention_probs)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_layer)
  context_layer = context_layer / tf.cast(
      tf.transpose(attention_sum, [0, 2, 1, 3]), context_layer.dtype)

  if num_partitions > 1:
    # partition along the heads dimension
    context_layer = xla_sharding.split(
        context_layer, 2, num_partitions, use_sharding_op=True)

  # split mask and scaling ops in dropout
  # move the scaling from dropout to here to save same mul ops
  # TODO(yuemmawang) automate this optimization in xla
  keep_prob = 1 - attention_probs_dropout_prob
  scale = 1 / keep_prob
  context_layer = tf.multiply(context_layer, scale)
  return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      num_partitions=1):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    num_partitions: (optional) int32. Number of SPMD partitions.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  prev_output = input_tensor
  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_common", reuse=tf.AUTO_REUSE):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          attention_output = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              layer_idx=layer_idx,
              total_layers=num_hidden_layers,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              num_partitions=num_partitions)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = dense_layer_3d_proj(
              attention_output, layer_idx, num_hidden_layers, hidden_size,
              num_attention_heads, attention_head_size,
              create_initializer(initializer_range), None, "dense")
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input,
                                        layer_idx, num_hidden_layers)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = dense_layer_2d(
            attention_output, layer_idx, num_hidden_layers, intermediate_size,
            create_initializer(initializer_range), intermediate_act_fn, "dense")

      if num_partitions > 1:
        # partition along the feature dimension
        intermediate_output = xla_sharding.split(
            intermediate_output, 2, num_partitions, use_sharding_op=True)

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = dense_layer_2d(intermediate_output, layer_idx,
                                      num_hidden_layers, hidden_size,
                                      create_initializer(initializer_range),
                                      None, "dense")
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output, layer_idx,
                                  num_hidden_layers)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    return all_layer_outputs
  else:
    return all_layer_outputs[-1]


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x.name,
                       x.device, cast_x.device)
  return cast_x
