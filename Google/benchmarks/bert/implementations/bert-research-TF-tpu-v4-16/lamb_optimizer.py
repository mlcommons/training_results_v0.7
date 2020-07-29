"""LAMB (Layer-wise Adaptive Moments) optimizer as TF1 tf.train.Optimizer.

See paper [Large Batch Optimization for Deep Learning: Training BERT in 76
minutes](https://arxiv.org/abs/1904.00962).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow.compat.v1 as tf


class LAMBOptimizer(tf.train.Optimizer):
  """Optimizer that implements the LAMBOptimizer as tf.train.Optimizer."""

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               weight_decay_rate=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               use_locking=False,
               name="LAMB"):
    super(LAMBOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta_1
    self._beta2 = beta_2
    self._epsilon = epsilon
    self._weight_decay_rate = weight_decay_rate
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None
    self._weight_decay_rate_t = None

  def _get_beta_accumulators(self):
    with tf.init_scope():
      graph = tf.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)
    weight_decay_rate = self._call_if_callable(self._weight_decay_rate)

    self._lr_t = tf.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = tf.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = tf.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = tf.convert_to_tensor(epsilon, name="epsilon")
    self._weight_decay_rate_t = tf.convert_to_tensor(
        weight_decay_rate, name="weight_decay_rate")

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = tf.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = tf.cast(beta2_power, var.dtype.base_dtype)
    lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = tf.cast(self._weight_decay_rate_t,
                                  var.dtype.base_dtype)
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = m * beta1_t + m_scaled_g_values
    m_t = tf.assign(m, m_t, use_locking=self._use_locking)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = v * beta2_t + v_scaled_g_values
    v_t = tf.assign(v, v_t, use_locking=self._use_locking)

    # ==== The following is with m_t_hat and v_t_hat
    m_t_hat = m_t / (1. - beta1_power)
    v_t_hat = v_t / (1. - beta2_power)

    v_sqrt = tf.sqrt(v_t_hat)
    update = m_t_hat / (v_sqrt + epsilon_t)

    # ==== The following is the original LAMBOptimizer implementation
    # v_sqrt = tf.sqrt(v_t_hat)
    # update = m_t / (v_sqrt + epsilon_t)

    var_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(var_name):
      update += weight_decay_rate_t * var

    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      if var.shape.ndims > 1 and var.shape[0] == 24:
        w_norm = tf.norm(var, 2, range(1, var.shape.ndims), True)
        g_norm = tf.norm(update, 2, range(1, var.shape.ndims), True)
      else:
        w_norm = tf.norm(var, ord=2)
        g_norm = tf.norm(update, ord=2)
      ratio = tf.where(
          tf.greater(w_norm, 0),
          tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

    var_update = var - ratio * lr_t * update
    return tf.assign(var, var_update, use_locking=self._use_locking).op

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = tf.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = tf.cast(beta2_power, var.dtype.base_dtype)
    lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = tf.cast(self._weight_decay_rate_t,
                                  var.dtype.base_dtype)
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = tf.assign(v, v * beta2_t, use_locking=self._use_locking)
    with tf.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    # ==== The following is with m_t_hat and v_t_hat
    m_t_hat = m_t / (1. - beta1_power)
    v_t_hat = v_t / (1. - beta2_power)

    v_sqrt = tf.sqrt(v_t_hat)
    update = m_t_hat / (v_sqrt + epsilon_t)

    # ==== The following is the original LAMBOptimizer implementation
    # v_sqrt = tf.sqrt(v_t_hat)
    # update = m_t / (v_sqrt + epsilon_t)

    var_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(var_name):
      update += weight_decay_rate_t * var

    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      w_norm = tf.norm(var, ord=2)
      g_norm = tf.norm(update, ord=2)
      ratio = tf.where(
          tf.greater(w_norm, 0),
          tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)
    var_update = tf.assign_sub(
        var, ratio * lr_t * update, use_locking=self._use_locking)
    return tf.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: tf.scatter_add(  # pylint: disable=g-long-lambda
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with tf.control_dependencies([tf.scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with tf.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with tf.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return tf.group(*update_ops + [update_beta1, update_beta2], name=name_scope)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
