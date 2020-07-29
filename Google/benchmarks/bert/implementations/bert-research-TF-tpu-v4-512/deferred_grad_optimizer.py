"""Optimizer wrapper for deferred gradient application."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

import REDACTED
from six.moves import zip
import tensorflow.compat.v1 as tf

from typing import Any, List


class GradientAggregationOptimizer(tf.train.Optimizer):
  """Optimizer wrapper providing deferred gradient application.

  Large hardware configurations are in high-demand, and difficult to come by.
  This class enables simulating execution on large hardware configurations, by
  accumulating gradients across multiple batches. A few caveats apply:

  * Batch statistics (e.g. batch norm) will continue to be based on the
   "micro-batch" size.
  * This effectively trades off computation/time: simulating a large cluster on
    a single core will take an excessive amount of time.

  N.B. Learning rate schedules may need to be adjusted in addition to using
  this optimizer. Schedules should either be scaled down by the relative batch
  size, or use a schedule based on the number of examples to be consistent
  across different batch sizes.
  """

  def __init__(self, opt: tf.train.Optimizer, grad_steps: int):
    self._opt = opt
    self._grad_steps = grad_steps
    self._counter = None

  def _create_slots(self, var_list: List[Any]):
    if not self._counter:
      self._counter = tf.get_variable(
          shape=[], initializer=tf.zeros_initializer, name='update_count')

    for v in var_list:
      self._opt._zeros_slot(v, 'grad_accum', 'GradientAccumulator')  # pylint: disable=protected-access

  def compute_gradients(self, loss, var_list: List[Any] = None, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    with tf.init_scope():
      self._create_slots([v for (_, v) in grads_and_vars])

    accums = []
    variables = []

    for g, v in grads_and_vars:
      accum = self.get_slot(v, 'grad_accum')
      variables.append(v)
      if isinstance(g, tf.IndexedSlices):
        scaled_grad = tf.IndexedSlices(
            g.values / self._grad_steps, g.indices, dense_shape=g.dense_shape)
        accums.append(accum.assign_add(scaled_grad))  # pytype: disable=attribute-error
      else:
        accums.append(accum.assign_add(g / self._grad_steps))  # pytype: disable=attribute-error

    def _apply_and_zero():
      apply_op = self._opt.apply_gradients(list(zip(accums, variables)))
      with tf.control_dependencies([apply_op]):
        zero_op = [tf.assign(accum, tf.zeros_like(accum)) for accum in accums]
      return tf.group(zero_op, tf.assign_add(self._counter, 1))

    def _accum():
      return tf.group(accums)

    accum_step = tf.cond(
        tf.equal(tf.mod(global_step, self._grad_steps), self._grad_steps - 1),
        _apply_and_zero, _accum)

    with tf.control_dependencies([accum_step]):
      global_step = tf.assign_add(global_step, 1)
      return tf.group(global_step)

  def get_slot(self, *args, **kwargs):
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    return self._opt.variables()
