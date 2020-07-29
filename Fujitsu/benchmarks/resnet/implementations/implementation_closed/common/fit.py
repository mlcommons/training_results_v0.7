# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" example train fit utility """
import logging
import os
import time
import re
import math
import mxnet as mx
import horovod.mxnet as hvd
import numpy as np

#### imports needed for fit monkeypatch
from mxnet.initializer import Uniform
from mxnet.context import cpu
from mxnet.monitor import Monitor
from mxnet.model import BatchEndParam
from mxnet.initializer import Uniform
from mxnet.io import DataDesc, DataIter, DataBatch
from mxnet.base import _as_list
import copy
##### imports needed for custom optimizer
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
                           multi_sum_sq, multi_lars)
from mxnet.ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                           mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update,
                           signsgd_update, signum_update,
                           multi_sgd_update, multi_sgd_mom_update, multi_mp_sgd_update,
                           multi_mp_sgd_mom_update,
                           lars_multi_sgd_update, lars_multi_sgd_mom_update,
                           lars_multi_mp_sgd_update, lars_multi_mp_sgd_mom_update)
from mxnet.ndarray import sparse
#####

from mlperf_logging.mllog import constants as mlperf_constants
from mlperf_log_utils import mx_resnet_print, all_reduce, mpiwrapper

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

@register
class SGDwLARS(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of grad is ``row_sparse`` and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * (rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row])
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
        state = momentum * state + rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    lazy_update : bool, optional
        Default is True. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.::

            False: results in using the same precision as the weights (default),
            True: makes internal 32-bit copy of the weights and applies gradients
            in 32-bit precision even if actual weights used in the model have lower precision.
            Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, lars=True, lars_eta=0.001, lars_eps=0, **kwargs):
        super(SGDwLARS, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))
        self.lars = lars
        self.lars_eta = lars_eta
        self.lars_eps = lars_eps
        self.skip = 0
        self.last_lr = None
        self.cur_lr = None


    def _get_lrs(self, indices):
        """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
        if self.cur_lr is not None:
            self.last_lr = self.cur_lr

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if self.cur_lr is None:
            self.last_lr = lr
        self.cur_lr = lr

        lrs = [lr for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                lrs[i] *= self.param_dict[index].lr_mult
            elif index in self.lr_mult:
                lrs[i] *= self.lr_mult[index]
            elif index in self.idx2name:
                lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lrs

    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            is_weight = n.endswith('_weight')
            is_fc_bias = 'fc' in n and 'bias' in n
            if not (is_weight or is_fc_bias):
                if hvd.rank() == 0:
                    print("skipping wd on {}".format(n))
                self.wd_mult[n] = 0.0
            else:
                if hvd.rank() == 0:
                    print("using wd on {}".format(n))

        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _l2norm(self, v, rescale=False):
        "L2 Norm implementation"
        v = v.astype('float32')
        if rescale:
            v *= self.rescale_grad
        norm = mx.nd.norm(v).asnumpy()[0]
        return norm

    def _get_lars(self, i, weight, g, lr, wd):
        "Returns a scaling factor for the learning rate for this layer"
        name = self.idx2name[i] if i in self.idx2name else str(i)
        if name.endswith('gamma') or name.endswith('beta') or name.endswith('bias'):
            return lr

        w_norm = self._l2norm(weight)
        g_norm = self._l2norm(g, rescale=True)

        if w_norm > 0.0 and g_norm > 0.0:
            lars = self.lars_eta * w_norm/(g_norm + wd * w_norm + self.lars_eps)
        else:
            lars = 1.0

        return lars * lr

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)

        wds = self._get_wds(indices)

        if self.lars:
            lrs = [self._get_lars(i, w, g, lr, wd) for (i, w, g, lr, wd) in zip(indices, weights, grads, lrs, wds)]

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum * (self.cur_lr / self.last_lr)

        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            current_index = 0
            while current_index < len(indices):
                sidx = current_index
                eidx = current_index + self.aggregate_num
                if not multi_precision:
                    if self.momentum > 0:
                        multi_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                grads[sidx:eidx],
                                                                states[sidx:eidx])),
                                             out=weights[sidx:eidx],
                                             num_weights=len(weights[sidx:eidx]),
                                             lrs=lrs[sidx:eidx],
                                             wds=wds[sidx:eidx],
                                             **kwargs)
                    else:
                        multi_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                            grads[sidx:eidx])),
                                         out=weights[sidx:eidx],
                                         num_weights=len(weights[sidx:eidx]),
                                         lrs=lrs[sidx:eidx],
                                         wds=wds[sidx:eidx],
                                         **kwargs)
                else:
                    if self.momentum > 0:
                        multi_mp_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                   grads[sidx:eidx],
                                                                   *zip(*states[sidx:eidx]))),
                                                out=weights[sidx:eidx],
                                                num_weights=len(weights[sidx:eidx]),
                                                lrs=lrs[sidx:eidx],
                                                wds=wds[sidx:eidx],
                                                **kwargs)
                    else:
                        multi_mp_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                               grads[sidx:eidx],
                                                               list(zip(*states[sidx:eidx]))[1])),
                                            out=weights[sidx:eidx],
                                            num_weights=len(weights[sidx:eidx]),
                                            lrs=lrs[sidx:eidx],
                                            wds=wds[sidx:eidx],
                                            **kwargs)
                current_index += self.aggregate_num
        else:
            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    if state is not None:
                        sgd_mom_update(weight, grad, state, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                                   lr=lr, wd=wd, **kwargs)
                else:
                    if state[0] is not None:
                        mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                          lr=lr, wd=wd, **kwargs)
                    else:
                        mp_sgd_update(weight, grad, state[1], out=weight,
                                      lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == np.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == np.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

@register
class SGDwFASTLARS(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of grad is ``row_sparse`` and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * (rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row])
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
        state = momentum * state + rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    lazy_update : bool, optional
        Default is True. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.::

            False: results in using the same precision as the weights (default),
            True: makes internal 32-bit copy of the weights and applies gradients
            in 32-bit precision even if actual weights used in the model have lower precision.
            Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, lars=True, lars_eta=0.001, lars_eps=0, **kwargs):
        super(SGDwFASTLARS, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))
        self.lars = lars
        self.lars_eta = lars_eta
        self.lars_eps = lars_eps
        self.skip = 0
        self.last_lr = None
        self.cur_lr = None


    def _get_lrs(self, indices):
        """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
        if self.cur_lr is not None:
            self.last_lr = self.cur_lr

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if self.cur_lr is None:
            self.last_lr = lr
        self.cur_lr = lr

        lrs = [lr for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                lrs[i] *= self.param_dict[index].lr_mult
            elif index in self.lr_mult:
                lrs[i] *= self.lr_mult[index]
            elif index in self.idx2name:
                lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lrs

    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            is_weight = n.endswith('_weight')
            is_fc_bias = 'fc' in n and 'bias' in n
            if not (is_weight or is_fc_bias):
                if hvd.rank() == 0:
                    print("skipping wd on {}".format(n))
                self.wd_mult[n] = 0.0
            else:
                if hvd.rank() == 0:
                    print("using wd on {}".format(n))

        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _l2norm(self, v, rescale=False):
        "L2 Norm implementation"
        v = v.astype('float32')
        if rescale:
            v *= self.rescale_grad
        norm = mx.nd.norm(v).asnumpy()[0]
        return norm

    def _get_lars(self, i, weight, g, lr, wd):
        "Returns a scaling factor for the learning rate for this layer"
        name = self.idx2name[i] if i in self.idx2name else str(i)
        if name.endswith('gamma') or name.endswith('beta') or name.endswith('bias'):
            return lr

        w_norm = self._l2norm(weight)
        g_norm = self._l2norm(g, rescale=True)

        if w_norm > 0.0 and g_norm > 0.0:
            lars = self.lars_eta * w_norm/(g_norm + wd * w_norm + self.lars_eps)
        else:
            lars = 1.0

        return lars * lr

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum * (self.cur_lr / self.last_lr)

        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            nb_params = len(indices)
            names = [self.idx2name[i] if i in self.idx2name else str(i) for i in indices]
            lars_idx = [i for i in range(nb_params) if not(names[i].endswith('gamma')
                        or names[i].endswith('beta') or names[i].endswith('bias'))]
            if self.lars and len(lars_idx) > 0:
                nb_lars = len(lars_idx)
                no_lars_idx = [i for i in range(nb_params) if (names[i].endswith('gamma') or
                               names[i].endswith('beta') or names[i].endswith('bias'))]
                cur_ctx = weights[0].context
                full_idx = lars_idx + no_lars_idx
                new_lrs = array([lrs[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
                new_wds = array([wds[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
                new_weights = [weights[i] for i in full_idx]
                new_grads = [grads[i] for i in full_idx]
                w_sum_sq = multi_sum_sq(*new_weights[:nb_lars], num_arrays=nb_lars)
                g_sum_sq = multi_sum_sq(*new_grads[:nb_lars], num_arrays=nb_lars)
                multi_lars(new_lrs[:nb_lars], w_sum_sq, g_sum_sq, new_wds[:nb_lars],
                           eta=self.lars_eta, eps=self.lars_eps, rescale_grad=self.rescale_grad,
                           out=new_lrs[:nb_lars])
                new_states = [states[i] for i in full_idx]
                # Same than usual using preloaded sgd functions
                sidx = 0
                while sidx < len(indices):
                    eidx = sidx + len(new_weights[sidx:sidx+self.aggregate_num])
                    if not multi_precision:
                        if self.momentum > 0:
                            lars_multi_sgd_mom_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           new_states[sidx:eidx])),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            lars_multi_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                            new_grads[sidx:eidx])),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    else:
                        if self.momentum > 0:
                            lars_multi_mp_sgd_mom_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           *zip(*new_states[sidx:eidx]))),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            lars_multi_mp_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           list(zip(*new_states[sidx:eidx]))[1])),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    sidx += self.aggregate_num
            else:
                current_index = 0
                while current_index < len(indices):
                    sidx = current_index
                    eidx = current_index + self.aggregate_num
                    if not multi_precision:
                        if self.momentum > 0:
                            multi_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                    grads[sidx:eidx],
                                                                    states[sidx:eidx])),
                                                 out=weights[sidx:eidx],
                                                 num_weights=len(weights[sidx:eidx]),
                                                 lrs=lrs[sidx:eidx],
                                                 wds=wds[sidx:eidx],
                                                 **kwargs)
                        else:
                            multi_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                grads[sidx:eidx])),
                                             out=weights[sidx:eidx],
                                             num_weights=len(weights[sidx:eidx]),
                                             lrs=lrs[sidx:eidx],
                                             wds=wds[sidx:eidx],
                                             **kwargs)
                    else:
                        if self.momentum > 0:
                            multi_mp_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                       grads[sidx:eidx],
                                                                       *zip(*states[sidx:eidx]))),
                                                    out=weights[sidx:eidx],
                                                    num_weights=len(weights[sidx:eidx]),
                                                    lrs=lrs[sidx:eidx],
                                                    wds=wds[sidx:eidx],
                                                    **kwargs)
                        else:
                            multi_mp_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                   grads[sidx:eidx],
                                                                   list(zip(*states[sidx:eidx]))[1])),
                                                out=weights[sidx:eidx],
                                                num_weights=len(weights[sidx:eidx]),
                                                lrs=lrs[sidx:eidx],
                                                wds=wds[sidx:eidx],
                                                **kwargs)
                    current_index += self.aggregate_num
        else:
            if self.lars:
                lrs = [self._get_lars(i, w, g, lr, wd) for (i, w, g, lr, wd) in
                       zip(indices, weights, grads, lrs, wds)]

            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    if state is not None:
                        sgd_mom_update(weight, grad, state, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                                   lr=lr, wd=wd, **kwargs)
                else:
                    if state[0] is not None:
                        mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                          lr=lr, wd=wd, **kwargs)
                    else:
                        mp_sgd_update(weight, grad, state[1], out=weight,
                                      lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == np.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == np.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

def get_epoch_size(args, kv):
    num_workers = hvd.size() if 'horovod' in args.kv_store else kv.num_workers
    return math.ceil(int(args.num_examples / num_workers) / args.batch_size)

def _get_gpu(gpus):
    idx = hvd.local_rank()
    gpu = gpus.split(",")[idx]
    return gpu

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = 0
    mx_resnet_print(key='lars_opt_base_learning_rate', val=args.lr)
    mx_resnet_print(key='lars_opt_learning_rate_warmup_epochs', val=args.warmup_epochs)
    if 'pow' in args.lr_step_epochs:
        if 'horovod' in args.kv_store:
            num_workers = hvd.size()
        else:
            num_workers = kv.num_workers if kv else 1
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        warmup_steps = epoch_size * args.warmup_epochs
        total_steps = epoch_size * args.num_epochs
        mx_resnet_print(key=mlperf_constants.LARS_OPT_LR_DECAY_STEPS,
                        val=args.num_epochs)
        return (args.lr, PolySchedule(args.lr, total_steps, warmup_steps))

    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        if 'horovod' in args.kv_store:
            num_workers = hvd.size()
        else:
            num_workers = kv.num_workers if kv else 1
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        mx_resnet_print(key=mlperf_constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                        val=step_epochs)
        mx_resnet_print(key=mlperf_constants.OPT_LR_DECAY_BOUNDARY_STEPS,
                        val=[lr * (args.lr_factor ** i) for i in range(len(step_epochs))])
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor,
                                                         base_lr=args.lr, warmup_steps=epoch_size * args.warmup_epochs,
                                                         warmup_mode=args.warmup_strategy))
    else:
        return (lr, None)

class PolySchedule():
    def __init__(self, base_lr, iterations, warmup_iterations):
        self.base_lr = base_lr
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.end_lr = 0.0001
        mx_resnet_print(key=mlperf_constants.LARS_OPT_LR_DECAY_POLY_POWER, val=2)
        mx_resnet_print(key=mlperf_constants.LARS_OPT_END_LR, val=self.end_lr)

    def __call__(self, iteration):
        if iteration <= self.warmup_iterations:
            return self.base_lr * (iteration / self.warmup_iterations)
        else:
            polyit = iteration - self.warmup_iterations
            polytotal = self.iterations - self.warmup_iterations

            return self.end_lr + ((self.base_lr - self.end_lr) * (1 - (polyit / polytotal))**2)

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--label-smoothing', type=float, default=0.0)
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--lars-eps', type=float, default=0,
                       help='lars epsilon param')
    train.add_argument('--lars-eta', type=float, default=0.001,
                       help='lars trust_factor param')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    train.add_argument('--eval-period', type=int, default=1, help='evaluation every N epochs')
    train.add_argument('--eval-offset', type=int, default=0, help='first evaluation on epoch N')

    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    # additional parameters for large batch sgd
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--logging-dir', type=str, default='logs')
    train.add_argument('--log', type=str, default='')
    train.add_argument('--bn-gamma-init0', action='store_true')
    train.add_argument('--epoch-size',type=int, default=0,
                       help='set number of batches in an epoch. useful for debugging')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    train.add_argument('--accuracy-threshold', default=1.0, type=float,
                       help='stop training after top1 reaches this value')
    return train


class CorrectCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='correct-count',
                 output_names=None, label_names=None):
        super(CorrectCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.sum_metric)

    def get_global(self):
        return (self.name, self.global_sum_metric)


class TotalCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='total-count',
                 output_names=None, label_names=None):
        super(TotalCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.num_inst)

    def get_global(self):
        return (self.name, self.global_num_inst)


class TopKCorrectCount(mx.metric.TopKAccuracy):
    def __init__(self, name='top-k-correct-count',
                 output_names=None, label_names=None):
        super(TopKCorrectCount, self).__init__(
                name=name, top_k=5,
                output_names=output_names, label_names=label_names)

    def get(self):
        return (self.name, self.sum_metric)

    def get_global(self):
        return (self.name, self.global_sum_metric)


class CrossEntropyCount(mx.metric.CrossEntropy):
    def __init__(self, name='cross-entropy',
                 output_names=None, label_names=None):
        super(CrossEntropyCount, self).__init__(
                name=name, output_names=output_names, label_names=label_names)

    def get(self):
        return (self.name, self.sum_metric)

    def get_global(self):
        return (self.name, self.global_sum_metric)



def mlperf_fit(self, args, train_data, eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None, kvstore='local',
               optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
               eval_end_callback=None,
               eval_batch_end_callback=None, initializer=Uniform(0.01),
               arg_params=None, aux_params=None, allow_missing=False,
               force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
               validation_metric=None, monitor=None, sparse_row_id_fn=None,
               eval_offset=0, eval_period=1,
               accuracy_threshold=1.0,
               multi_gpu_per_process=False):

    assert num_epoch is not None, 'please specify number of epochs'

    self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
              for_training=True, force_rebind=force_rebind)

    if monitor is not None:
        self.install_monitor(monitor)

    self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                     allow_missing=allow_missing, force_init=force_init)
    self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                        optimizer_params=optimizer_params)

    if validation_metric is None:
        validation_metric = eval_metric
    ###########################################################################
    # Adding Correct and Total Count metrics
    ###########################################################################
    if not isinstance(validation_metric, list):
        validation_metric = [validation_metric]

    validation_metric = mx.metric.create(validation_metric)

    if not isinstance(validation_metric, mx.metric.CompositeEvalMetric):
        vm = mx.metric.CompositeEvalMetric()
        vm.append(validation_metric)
        validation_metric = vm

    for m in [CorrectCount(), TotalCount()]:
        validation_metric.metrics.append(m)
    ###########################################################################

    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)
    block_epoch_start = begin_epoch
    block_epoch_count = eval_offset + 1 - (begin_epoch % eval_period)
    if block_epoch_count < 0:
        block_epoch_count += eval_period
    mx_resnet_print(key=mlperf_constants.BLOCK_START,
        metadata={'first_epoch_num': block_epoch_start + 1, 'epoch_count': block_epoch_count})
    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        mx_resnet_print(key=mlperf_constants.EPOCH_START, metadata={'epoch_num': epoch + 1})
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            if monitor is not None:
                monitor.tic()
            self.forward_backward(data_batch)
            self.update()

            if isinstance(data_batch, list):
                self.update_metric(eval_metric,
                                   [db.label for db in data_batch],
                                   pre_sliced=True)
            else:
                self.update_metric(eval_metric, data_batch.label)

            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
            except StopIteration:
                end_of_batch = True

            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)

            nbatch += 1
        mx_resnet_print(key=mlperf_constants.EPOCH_STOP, metadata={"epoch_num": epoch + 1})
        # one epoch of training is finished
        toc = time.time()
        if kvstore:
            if kvstore.rank == 0:
                self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
        elif 'horovod' in args.kv_store:
            if hvd.rank() == 0:
                self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
        else:
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # sync aux params across devices if there is more than one GPU per process
        if multi_gpu_per_process:
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, self.symbol, arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data and epoch % eval_period == eval_offset:
            mx_resnet_print(key=mlperf_constants.EVAL_START, metadata={'epoch_num': epoch + 1})
            res = self.score(eval_data, validation_metric,
                             score_end_callback=eval_end_callback,
                             batch_end_callback=eval_batch_end_callback, epoch=epoch)

            if kvstore:
                if kvstore.rank == 0:
                    for name, val in res:
                        self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            elif 'horovod' in args.kv_store:
                if hvd.rank() == 0:
                    for name, val in res:
                        self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            else:
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            res = dict(res)

            acc = [res['correct-count'], res['total-count']]
            acc = all_reduce(acc)
            acc = acc[0]/acc[1]
            mx_resnet_print(key=mlperf_constants.EVAL_STOP, metadata={'epoch_num': epoch + 1})

            mx_resnet_print(key=mlperf_constants.EVAL_ACCURACY, val=acc,
                            metadata={'epoch_num': epoch + 1})


            mx_resnet_print(key=mlperf_constants.BLOCK_STOP,
                            metadata={'first_epoch_num': block_epoch_start + 1})
            if acc > accuracy_threshold:
                mx_resnet_print(key=mlperf_constants.RUN_STOP,
                                metadata={'status': 'success'})
                return epoch

            if epoch < (num_epoch - 1):
                block_epoch_start = epoch + 1
                block_epoch_count = num_epoch - epoch - 1
                if block_epoch_count > eval_period:
                    block_epoch_count = eval_period
                mx_resnet_print(key=mlperf_constants.BLOCK_START,
                    metadata={'first_epoch_num': block_epoch_start + 1,
                              'epoch_count': block_epoch_count})
        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()
    mx_resnet_print(key=mlperf_constants.RUN_STOP,
                    metadata={'status': 'aborted'})
    return num_epoch

def fit(args, kv, model, initializer, data_loader, devs, arg_params, aux_params, **kwargs):
    """
    train a model
    args : argparse returns
    model : loaded model of the neural network
    initializer : weight initializer
    data_loader : function that returns the train and val data iterators
    devs : devices for training
    arg_params : model parameters
    aux_params : model parameters
    """
    if 'horovod' in args.kv_store:
        kv = None
        rank = hvd.rank()
        num_workers = hvd.size()
    else:
        rank = kv.rank
        num_workers = kv.num_workers
    if args.profile_server_suffix:
        mx.profiler.set_config(filename=args.profile_server_suffix, profile_all=True, profile_process='server')
        mx.profiler.set_state(state='run', profile_process='server')

    if args.profile_worker_suffix:
        if num_workers > 1:
            filename = 'rank' + str(rank) + '_' + args.profile_worker_suffix
        else:
            filename = args.profile_worker_suffix
        mx.profiler.set_config(filename=filename, profile_all=True, profile_process='worker')
        mx.profiler.set_state(state='run', profile_process='worker')

    # logging
    head = '%(asctime)-15s Node[' + str(rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    epoch_size = get_epoch_size(args, kv)

    # data iterators
    (train, val) = data_loader(args, kv)
    if 'dist' in args.kv_store and not 'async' in args.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        if not args.use_dali:
            train = mx.io.ResizeIter(train, epoch_size)

    # save model
    epoch_end_callbacks = []

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    if 'horovod' in args.kv_store:
        optimizer_params['rescale_grad'] = 1. / args.batch_size

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd', 'sgdwlars', 'sgdwfastlars'}
    mx_resnet_print(key='lars_opt_weight_decay', val=args.wd)
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom
        mx_resnet_print(key='lars_opt_momentum', val=args.mom)

    if args.optimizer in {'sgdwlars', 'sgdwfastlars'}:
        optimizer_params['lars'] = True
        optimizer_params['lars_eta'] = args.lars_eta
        optimizer_params['lars_eps'] = args.lars_eps
        mx_resnet_print(key=mlperf_constants.OPT_NAME,
                        val='lars')
        mx_resnet_print(key=mlperf_constants.LARS_EPSILON,
                        val=args.lars_eps)
    else:
        mx_resnet_print(
            key=mlperf_constants.OPT_NAME,
            val='sgd')

    if 'horovod' in args.kv_store:
        # Setting idx2name dictionary, required to mask out entries for weight decay.
        idx2name = {}
        for i,n in enumerate(model._exec_group.param_names):
            idx2name[i] = n

        opt = mx.optimizer.create(args.optimizer, sym=None, param_idx2name=idx2name, **optimizer_params)
        # Horovod: wrap optimizer with DistributedOptimizer
        # Note: enabling skip_average in DistributedOptimizer. Normalization is baked into rescale_grad.
        opt = hvd.DistributedOptimizer(opt)
    else:
        opt = args.optimizer

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = []
    if 'horovod' in args.kv_store:
        # if using horovod, only report on rank 0 with global batch size
        if rank == 0:
            batch_end_callbacks.append(mx.callback.Speedometer(
                num_workers*args.batch_size, args.disp_batches))
        mx_resnet_print(key=mlperf_constants.GLOBAL_BATCH_SIZE,
                        val=num_workers * args.batch_size)
    else:
        batch_end_callbacks.append(mx.callback.Speedometer(
            args.batch_size, args.disp_batches))
        mx_resnet_print(key=mlperf_constants.GLOBAL_BATCH_SIZE,
                        val=args.batch_size)

    # run
    last_epoch = mlperf_fit(model,
                            args,
                            train,
                            begin_epoch=0,
                            num_epoch=args.num_epochs,
                            eval_data=val,
                            eval_metric=eval_metrics,
                            kvstore=kv,
                            optimizer=opt,
                            optimizer_params=optimizer_params,
                            initializer=None if 'horovod' in args.kv_store else initializer,
                            arg_params=arg_params,
                            aux_params=aux_params,
                            batch_end_callback=batch_end_callbacks,
                            epoch_end_callback=epoch_end_callbacks, #checkpoint if args.use_dali else ,,
                            allow_missing=True, 
                            eval_offset=args.eval_offset,
                            eval_period=args.eval_period,
                            accuracy_threshold=args.accuracy_threshold,
                            multi_gpu_per_process=(len(devs) > 1),
                            monitor=None)

    if args.profile_server_suffix:
        mx.profiler.set_state(state='run', profile_process='server')
    if args.profile_worker_suffix:
        mx.profiler.set_state(state='run', profile_process='worker')
