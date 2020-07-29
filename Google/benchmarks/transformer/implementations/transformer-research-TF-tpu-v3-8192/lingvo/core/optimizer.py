# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.compat as tf
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import base_layer
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import py_utils
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import summary_utils


class Base(base_layer.BaseLayer):
  """Base class for all optimizers."""

  @classmethod
  def Params(cls):
    p = super(Base, cls).Params()
    p.name = cls.__name__
    p.Define(
        'use_bf16_gradients_ar', False,
        'Whether to use bfloat16 dtype for gradients all-reduce. '
        'This applies to TPU only.')
    return p

  def GetOptimizer(self, lr):
    """Returns the TF optimizer object."""
    raise NotImplementedError('Abstract method')

  def AddSummary(self, lr, optimizer, var_grad):
    """Adds summary if needed."""
    pass

  def ComputeGradients(self, loss, vmap, *args, **kwargs):
    """Allows subclasses control computation of gradients."""
    kwargs['use_bf16_gradients_ar'] = self.params.use_bf16_gradients_ar
    return py_utils.ComputeGradients(loss, vmap, *args, **kwargs)

  def VarReuseForSlotVars(self):
    """Multi-task models require AUTO_REUSE for var sharing."""
    var_reuse = False
    if py_utils.GetOpportunisticVariableReuse():
      var_reuse = tf.AUTO_REUSE
    return var_reuse

  def Apply(self, lr, var_grad):
    """Applies the gradient to the variable.

    Args:
      lr: A scalar. The base learning rate.
      var_grad: A `.NestedMap` of (var, grad) pairs.

    Returns:
      The variable update op.
    """
    optimizer = self.GetOptimizer(lr)

    def _Apply():
      if self.params.use_bf16_gradients_ar:
        return optimizer.apply_gradients(
            [(tf.cast(g, tf.float32), v) for (v, g) in var_grad.Flatten()],
            name='meta_backprop')
      else:
        return optimizer.apply_gradients(
            [(g, v) for (v, g) in var_grad.Flatten()], name='meta_backprop')

    if not py_utils.use_resource_variables():
      var_update_op = _Apply()
    else:
      # Many optimizers, e.g., Adam, Adagrad, etc., create
      # variables. We need to ensure name scope and variable scope are
      # cleared. Otherwise, tpu.batch_parallel does not work.
      with tf.name_scope(None):
        with tf.variable_scope(
            tf.VariableScope(
                use_resource=True, reuse=self.VarReuseForSlotVars())):
          var_update_op = _Apply()
    self.AddSummary(lr, optimizer, var_grad)
    return var_update_op

  def ApplyPostTrainingLoop(self, global_step):
    """Applies any computation to run after each tpu trainining loop.

    Args:
      global_step: Global step variable.

    Returns:
      Ops to run after training loop ends.
    """
    return tf.no_op()


class CompositeOptimizer(Base):
  """Composite Optimizer.

  A composite optimizer is composed of one or more Lingvo Optimizer objects
  where regex specifies which variables should use which optimizer. The
  optimizer_map dictionary must specify a default_optimizer regex to a
  (Lingvo Optimizer, learning rate) tuple which will be applied to all variables
  which do not match an earlier regex.

  For example,

  optimizer_map = {'a': Adam, 'b': Adagrad, 'default_optimizer': SGD}

  will apply Adam to all variables which contain an 'a' in their name, apply
  Adagrad to all variables which contain a 'b' in their name, and apply SGD to
  the variables which do not contain either 'a' or 'b'.

  If a non-default_optimizer matches more than one variable -- in this example
  variables with both 'a' and 'b' in their name -- an exception is thrown.
  """

  @classmethod
  def Params(cls):
    p = super(CompositeOptimizer, cls).Params()
    p.Define(
        'optimizer_map', None,
        'Mapping of variable regex to (Lingvo Optimizer, learning rate) tuple.')
    return p

  def __init__(self, params):
    super(CompositeOptimizer, self).__init__(params)
    self._optimizer_map = {}
    self._lr_map = {}
    for index, regex in enumerate(params.optimizer_map):
      sub_optimizer, learning_rate = params.optimizer_map[regex]
      self.CreateChild('sub_{}_{}'.format(sub_optimizer.name, index),
                       sub_optimizer)
      self._optimizer_map[regex] = self.children['sub_{}_{}'.format(
          sub_optimizer.name, index)]
      self._lr_map[regex] = learning_rate

    if 'default_optimizer' not in self._optimizer_map:
      raise KeyError('default_optimizer is not found in optimizer_map. Please '
                     'specify a default_optimizer regex and its associated '
                     '(Lingvo Optimizer, learning rate) tuple.')

  def GetOptimizer(self, lr):
    """Returns a dictionary of regex to TF optimizer objects."""
    return {
        k: v.GetOptimizer(self._lr_map[k])
        for k, v in self._optimizer_map.items()
    }

  def Apply(self, lr, var_grad):
    """For each optimizer, apply the gradient to the variable.

    Args:
      lr: A scalar. The base learning rate.
      var_grad: A `.NestedMap` of (var, grad) pairs.

    Returns:
      The variable update op.

    Raises:
      Exception: When the regex overlaps with or does not cover all variables.
    """
    # Override inherited GetOptimizer even though learning rate is unused.
    tf_optimizer_map = self.GetOptimizer(0)
    var_grad_map = {regex: [] for regex in self._optimizer_map}

    for (v, g) in var_grad.Flatten():
      regex_match = 0
      for regex in self._optimizer_map:
        if re.match(regex, v.name):
          var_grad_map[regex].append((g, v))
          regex_match += 1
      if regex_match == 0:
        var_grad_map['default_optimizer'].append((g, v))
      if regex_match > 1:
        raise Exception('Variable {} is matched {} times by regex {}'.format(
            v.name, regex_match, list(self._optimizer_map.keys())))

    def _Apply():
      """Use the matched optimizer to apply the gradients."""
      train_ops = []
      non_default_regex = [
          regex for regex in self._optimizer_map if regex != 'default_optimizer'
      ]
      for regex in self._optimizer_map:
        if var_grad_map[regex]:
          opt = tf_optimizer_map[regex]
          train_ops.append(opt.apply_gradients(var_grad_map[regex]))
          # pylint: disable=cell-var-from-loop, g-long-lambda
          if regex == 'default_optimizer':
            filtered_var_grad = var_grad.FilterKeyVal(lambda k, v: any(
                [re.match(i, v.var.name) for i in non_default_regex]))
          else:
            filtered_var_grad = var_grad.FilterKeyVal(
                lambda k, v: (re.match(regex, v.var.name)))
          # pylint: enable=cell-var-from-loop, g-long-lambda
          self._optimizer_map[regex].AddSummary(self._lr_map[regex], opt,
                                                filtered_var_grad)
      return tf.group(*train_ops, name='composite_optimizer_train_op')

    if not py_utils.use_resource_variables():
      var_update_op = _Apply()
    else:
      # Many optimizers, e.g., Adam, Adagrad, etc., create
      # variables. We need to ensure name scope and variable scope are
      # cleared. Otherwise, tpu.batch_parallel does not work.
      var_reuse = False
      if py_utils.GetOpportunisticVariableReuse():
        var_reuse = tf.AUTO_REUSE
      with tf.name_scope(None):
        with tf.variable_scope(
            tf.VariableScope(use_resource=True, reuse=var_reuse)):
          var_update_op = _Apply()
    return var_update_op

  def ApplyPostTrainingLoop(self, global_step):
    """Apply any computation to run after each tpu training loop for each optimizer.

    Args:
      global_step: Global step variable.

    Returns:
      Ops to run after training loop ends.
    """
    post_training_ops = [
        opt.ApplyPostTrainingLoop(global_step)
        for _, opt in self._optimizer_map.items()
    ]
    return tf.group(*post_training_ops)


class SGD(Base):
  """SGD."""

  def GetOptimizer(self, lr):
    return tf.train.GradientDescentOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('sgd_lr', lr)


class Momentum(Base):
  """Momentum optimizer."""

  @classmethod
  def Params(cls):
    p = super(Momentum, cls).Params()
    p.Define(
        'alpha', 0.9, 'The damping factor in the momentum '
        'optimizer. This controls how the velocity (averaged '
        'past gradients) is decayed over time.')
    p.Define('use_nesterov', False, 'True iff use Nesterov')
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.MomentumOptimizer(
        learning_rate=lr, momentum=p.alpha, use_nesterov=p.use_nesterov)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('momentum_lr', lr)


class RMSProp(Base):
  """RMSProp optimizer."""

  @classmethod
  def Params(cls):
    p = super(RMSProp, cls).Params()
    p.Define('decay', 0.9, 'Discounting factor for the history/coming gradient')
    p.Define('momentum', 0.9, 'Momentum in RMSProp.')
    p.Define(
        'epsilon', 1.0,
        'Epsilon term for RMSProp. Small value to avoid zero denominator.')
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.RMSPropOptimizer(
        lr, p.decay, momentum=p.momentum, epsilon=p.epsilon)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('rmsprop_lr', lr)


class Adagrad(Base):
  """Adagrad."""

  @classmethod
  def Params(cls):
    p = super(Adagrad, cls).Params()
    p.Define('initial_accumulator_value', 1.0,
             "Adagrad's initial_accumulator_value.")
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdagradOptimizer(
        learning_rate=lr, initial_accumulator_value=p.initial_accumulator_value)

  def AddSummary(self, lr, optimizer, var_grad):
    p = self.params
    summary_utils.scalar('adagrad_lr', lr)
    for v, _ in var_grad.Flatten():
      slot = optimizer.get_slot(v, 'accumulator')
      assert slot is not None
      summary_utils.scalar('optimizer/adagrad_accum_%s' % v.name,
                           tf.reduce_mean(slot))


class AdaDelta(Base):
  """AdaDelta optimizer."""

  @classmethod
  def Params(cls):
    p = super(AdaDelta, cls).Params()
    p.Define('decay', 0.95,
             'Discounting factor for the history/coming gradient')
    p.Define(
        'epsilon', 1e-8,
        'Epsilon term for AdaDelta. Small value to avoid zero denominator.')
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdadeltaOptimizer(
        learning_rate=lr, rho=p.decay, epsilon=p.epsilon)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adadelta_lr', lr)


class Adam(Base):
  """Adam."""

  @classmethod
  def Params(cls):
    p = super(Adam, cls).Params()
    p.Define('beta1', 0.9, 'Beta1 for Adam.')
    p.Define('beta2', 0.999, 'Beta2 for Adam.')
    p.Define('epsilon', 1e-6, 'Epsilon for Adam.')
    p.name = 'Adam'
    return p

  @staticmethod
  def ParamsA():
    """Convenient method for a commonly used Adam config."""
    return Adam.Params().Set(beta1=0.9, beta2=0.997, epsilon=1e-9)

  @staticmethod
  def ParamsB():
    """Convenient method for another commonly used Adam config."""
    return Adam.Params().Set(beta1=0.9, beta2=0.98, epsilon=1e-9)

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=p.beta1,
        beta2=p.beta2,
        epsilon=p.epsilon,
        name=p.name)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adam_lr', lr)


class Accumulator(Base):
  """Gradient accumulator wrapper."""

  @classmethod
  def Params(cls):
    p = super(Accumulator, cls).Params()
    p.Define('optimizer_tpl', Adam.Params(),
             'Params for the wrapped optimizer.')
    p.Define(
        'accum_steps', 5, 'Number of gradient accumulation steps'
        ' before invoking wrapped optimizer.')
    p.name = 'Accumulator'
    return p

  def __init__(self, params):
    super(Accumulator, self).__init__(params)
    p = self.params
    self.CreateChild('_opt', p.optimizer_tpl)

  def Apply(self, lr, var_grad):
    p = self.params

    def _Acc(vg):
      """Updating accumulators."""

      v, g = vg
      with tf.variable_scope(v.op.name):
        _, a = py_utils.CreateVariable(
            'grad_accumulator',
            py_utils.WeightParams(v.get_shape(),
                                  py_utils.WeightInit.Constant(0.0),
                                  self.params.dtype),
            trainable=False)
        a = tf.assign_add(a, g)

      return py_utils.VarGrad(v, a)

    var_grad = var_grad.Transform(_Acc)

    def _ApplyAndReset():
      with tf.control_dependencies([
          self._opt.Apply(
              lr, py_utils.ApplyGradMultiplier(var_grad, 1. / p.accum_steps))
      ]):
        return tf.group(
            *[tf.assign(a, tf.zeros_like(a)) for _, a in var_grad.Flatten()])

    return tf.cond(
        tf.equal(
            tf.math.floormod(self.theta.global_step, p.accum_steps),
            p.accum_steps - 1), _ApplyAndReset, lambda: tf.group(tf.no_op()))

  def GetOptimizer(self, lr):
    return self._opt.GetOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    return self._opt.AddSummary(lr, optimizer, var_grad)
