# Lint as: python3
"""Utility functions to load a Resnet34 tf checkpoint into JAX-ssd model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from flax import optim
import jax.numpy as jnp
import tensorflow.compat.v1 as tf


def get_tf_variables(resnet_checkpoint):
  """Load tensorflow parameter weights from a given tf checkpoint.

  Args:
    resnet_checkpoint: String denoting to the tf checkpoint location.
  Returns:
    A dictionary from parameter names to numpy tensors.
  """
  ckpt_reader = tf.train.load_checkpoint(resnet_checkpoint)
  var_list = tf.train.list_variables(resnet_checkpoint)
  tf_parameter_dict = {}
  for v in var_list:
    var_name = v[0]
    if ckpt_reader.has_tensor(var_name):
      # variable in checkpoint starts from 'resnet/' and get rid of this prefix.
      tf_parameter_dict[var_name[7:]] = ckpt_reader.get_tensor(var_name)
  return tf_parameter_dict


def _get_jax_to_tf_batch_norm_mapping(tf_bn_layer_name):
  """Returns a dictionary from JAX naming to tf naming for batch norm parameters.

  Args:
    tf_bn_layer_name: String denoting the name of the batch norm layer in tf.
  Returns:
    A dictionary from JAX naming to tf naming for batch norm parameters.
  """
  return {'bias': '%s/beta' % tf_bn_layer_name,
          'scale': '%s/gamma' % tf_bn_layer_name}


def _get_tf_momentum_var(tf_layer):
  """Returns the name of the momentum parameter for a given tf layer.

  Args:
    tf_layer: String for the name of the tf layer.
  Returns:
    String denoting the parameter name for the momentum parameter.
  """
  return '%s/Momentum' % tf_layer


def _get_tf_conv_kernel(conv_counter):
  """Returns the name of the conv kernel parameter for the given conv index.

  Args:
    conv_counter: Integer number denoting the conv index.
  Returns:
    A String denoting the name of the conv kernel parameter.
  """
  if conv_counter != 0:
    return 'conv2d_%d/kernel' % conv_counter
  return 'conv2d/kernel'


def _get_tf_batch_norm_parameter_name(bn_counter):
  """Returns the name of the batch_norm  layer for the given batch norm index.

  Args:
    bn_counter: Integer number denoting the batch norm op index.
  Returns:
    A String denoting the name of the batch norm layer.
  """
  if bn_counter != 0:
    return 'batch_normalization_%d' % bn_counter
  return 'batch_normalization'


def _update_jax_layer_weights_with_momentum(jax_parameter_dictionary,
                                            jax_momentum_dictionary,
                                            jax_parameter_name,
                                            tf_parameter_dictionary,
                                            tf_parameter_name):
  """Updates the weights on the JAX side for a given layer with momentum parameters.

  Args:
    jax_parameter_dictionary: JAX parameter dictionary from parameter names to
      parameter values (device numpy array).
    jax_momentum_dictionary: JAX parameter dictionary from parameter names to
      momentum parameter values (device numpy array).
    jax_parameter_name: String name of the JAX parameter name that will be
      updated.
    tf_parameter_dictionary: TF parameter dictionary from parameter names to
      their values (numpy array).
    tf_parameter_name: The corresponding tf parameter name of the JAX parameter.
  """

  jax_parameter_dictionary[jax_parameter_name] = jnp.reshape(
      jnp.array(tf_parameter_dictionary[tf_parameter_name]),
      jax_parameter_dictionary[jax_parameter_name].shape)

  tf_moment_var = _get_tf_momentum_var(tf_parameter_name)
  # Rewriting the momentum parameter requires access to private
  # _MomentumParamState
  # pylint: disable=protected-access
  jax_momentum_dictionary[jax_parameter_name] = (
      optim.momentum._MomentumParamState(jnp.reshape(
          jnp.array(tf_parameter_dictionary[tf_moment_var]),
          jax_momentum_dictionary[jax_parameter_name].momentum.shape)))
  # pylint: enable=protected-access


def _update_batch_norm_weights(jax_parameter_dictionary,
                               jax_momentum_dictionary,
                               jax_state_dict,
                               tf_parameter_dictionary,
                               tf_batch_norm_parameter_name):
  """Updates the weights on the JAX side for a given batch norm layer with momentum parameters.

  Also updates the moving mean and moving variances of the state.

  Args:
    jax_parameter_dictionary: JAX parameter dictionary from parameter names to
      parameter values (device numpy array).
    jax_momentum_dictionary: JAX parameter dictionary from parameter names to
      momentum parameter values (device numpy array).
    jax_state_dict: Result of state.as_dict() holding the moving mean and
      moving variance of the batch norm layers.
    tf_parameter_dictionary: TF parameter dictionary from parameter names to
      their values (numpy array).
    tf_batch_norm_parameter_name: The tf parameter name for the batch norm layer
      that will be updated.
  """
  jax_to_tf_bn = _get_jax_to_tf_batch_norm_mapping(
      tf_batch_norm_parameter_name)
  for jax_parameter_name, tf_parameter_name in jax_to_tf_bn.items():
    _update_jax_layer_weights_with_momentum(jax_parameter_dictionary,
                                            jax_momentum_dictionary,
                                            jax_parameter_name,
                                            tf_parameter_dictionary,
                                            tf_parameter_name)
  tf_variance = '%s/moving_variance' %  tf_batch_norm_parameter_name
  tf_mean = '%s/moving_mean' %  tf_batch_norm_parameter_name
  jax_state_dict['var'] = (
      jnp.reshape(
          jnp.array(tf_parameter_dictionary[tf_variance]),
          jax_state_dict['var'].shape))
  jax_state_dict['mean'] = (
      jnp.reshape(
          jnp.array(tf_parameter_dictionary[tf_mean]),
          jax_state_dict['mean'].shape))


def _update_conv_weights(jax_parameter_dictionary, jax_momentum_dictionary,
                         tf_parameter_dictionary,
                         tf_conv_parameter_name):
  """Updates the weights on the JAX side for a given conv layer with momentum parameters.

  Args:
    jax_parameter_dictionary: JAX parameter dictionary from parameter names to
      parameter values (device numpy array).
    jax_momentum_dictionary: JAX parameter dictionary from parameter names to
      momentum parameter values (device numpy array).
    tf_parameter_dictionary: TF parameter dictionary from parameter names to
      their values (numpy array).
    tf_conv_parameter_name: The tf parameter name for the conv layer
      that will be updated.
  """
  jax_conv_kernel_name = 'kernel'
  _update_jax_layer_weights_with_momentum(jax_parameter_dictionary,
                                          jax_momentum_dictionary,
                                          jax_conv_kernel_name,
                                          tf_parameter_dictionary,
                                          tf_conv_parameter_name)


def _to_flat_dict_key(keys):
  """Converts a list of nested keys to flat keys used in state.as_dict().

  Args:
    keys: List of keys from outmost to innermost.
  Returns:
    Corresponding flat dictionary for the given list of keys.
  """
  return '/' + '/'.join(keys)


def load_from_tf_checkpoints(jax_model, jax_model_state, params):
  """Loads the given resnet checkpoint and updates the ssd JAX model parameters accordingly.

  Args:
    jax_model: An instance of flax.optim.Optimizer. Output of
      optimizer.create(). Should not be replicated.
    jax_model_state: An instance of flax.nn.Collection, result of
      flax.nn.Module.create(). Should not be replicated.
    params: parameter dictionary.
  Returns:
    The updated jax_model and jax_model_state.
  """
  resnet_checkpoint = params['resnet_checkpoint']
  batch_norm_names = ('proj_bn', 'bn2', 'bn3')
  conv_names = ('proj_conv', 'conv1', 'conv2')
  jax_init_batch_norm_name = 'init_bn'
  jax_init_conv_name = 'init_conv'
  # There are 13 resnet layers used in SSD.
  num_layers = 13
  bn_counter = 0
  conv2d_counter = 0

  tf_parameters_dict = get_tf_variables(resnet_checkpoint)
  jax_state_dict = jax_model_state.as_dict()

  jax_resnet_parameters = jax_model.target.params['ResNet_0']
  jax_resnet_momentum_parameters = (
      jax_model.state.param_states.params['ResNet_0'])

  _update_batch_norm_weights(
      jax_resnet_parameters[jax_init_batch_norm_name],
      jax_resnet_momentum_parameters[jax_init_batch_norm_name],
      jax_state_dict[_to_flat_dict_key(['ResNet_0', jax_init_batch_norm_name])],
      tf_parameters_dict,
      _get_tf_batch_norm_parameter_name(bn_counter))
  bn_counter += 1

  _update_conv_weights(
      jax_resnet_parameters[jax_init_conv_name],
      jax_resnet_momentum_parameters[jax_init_conv_name],
      tf_parameters_dict,
      _get_tf_conv_kernel(conv2d_counter))

  conv2d_counter += 1

  for i in range(num_layers):
    str_i = str(i)
    jax_residual_block_params = jax_resnet_parameters['ResidualBlock_'+str_i]
    jax_residual_block_momentum_params = (
        jax_resnet_momentum_parameters['ResidualBlock_'+str_i])
    for bn in batch_norm_names:
      if bn in jax_residual_block_params:
        if bn not in jax_residual_block_momentum_params:
          raise ValueError(
              'Batch norm:%s cannot be found in momentum params.' % bn)
        _update_batch_norm_weights(
            jax_residual_block_params[bn],
            jax_residual_block_momentum_params[bn],
            jax_state_dict[
                _to_flat_dict_key(['ResNet_0', 'ResidualBlock_'+str_i, bn])],
            tf_parameters_dict, _get_tf_batch_norm_parameter_name(bn_counter))
        bn_counter += 1

    for conv in conv_names:
      if conv in jax_residual_block_params:
        if conv not in jax_residual_block_momentum_params:
          raise ValueError(
              'Conv:%s cannot be found in momentum params.' % conv)
        _update_conv_weights(jax_residual_block_params[conv],
                             jax_residual_block_momentum_params[conv],
                             tf_parameters_dict,
                             _get_tf_conv_kernel(conv2d_counter))
        conv2d_counter += 1
  return jax_model, jax_model_state
