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
                           preloaded_multi_sgd_update, preloaded_multi_sgd_mom_update,
                           preloaded_multi_mp_sgd_update, preloaded_multi_mp_sgd_mom_update)
from mxnet.ndarray import sparse
#####

from mlperf_log import mpiwrapper, allreduce
import mlperf_log as mll

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

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
                self.wd_mult[n] = 0.0

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
                            preloaded_multi_sgd_mom_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           new_states[sidx:eidx])),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            preloaded_multi_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                            new_grads[sidx:eidx])),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    else:
                        if self.momentum > 0:
                            preloaded_multi_mp_sgd_mom_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           *zip(*new_states[sidx:eidx]))),
                                        new_lrs[sidx:eidx],
                                        new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            preloaded_multi_mp_sgd_update(
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

def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, sparse_row_id_fn=None, accuracy_target=1.0,
            eval_frequency=1, eval_offset=0, logger=None):
    assert num_epoch is not None, 'please specify number of epochs'

    if 'horovod' in kvstore:
        rank = hvd.rank()
        local_rank = hvd.local_rank()
    else:
        rank = 0
        local_rank = 0
    
    profiler_on = os.getenv('RESNET50_PROFILING', False) and (rank == 0)
    if profiler_on:
        self.logger.info("Profiling is enabled")

    stop_iter = int(os.getenv('RESNET50_STOP_ITERATION', '0'))
    if stop_iter > 0:
        self.logger.info("Training will stop at iteration {} of the first epoch".format(stop_iter))

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
    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)

    block_epoch_start = begin_epoch
    block_epoch_count = eval_offset + 1 - (begin_epoch % eval_frequency)
    if block_epoch_count < 0:
        block_epoch_count += eval_frequency
    mll.block_start(block_epoch_start+1, count=block_epoch_count)

    if profiler_on:
        mx.profiler.set_config(profile_symbolic=True, profile_imperative=True, profile_memory=False,
                                profile_api=True, filename='resnet50_profile.json', aggregate_stats=True)
        mx.profiler.set_state('run')

    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        mll.epoch_start(epoch+1)
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        early_stop = False
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

            if end_of_batch:
                eval_name_vals = eval_metric.get_global_name_value()

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                    eval_metric=eval_metric,
                                                    locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            nbatch += 1
            if stop_iter > 0 and nbatch >= stop_iter:
                early_stop = True
                self.logger.info("Training stopped at {} iteration. Clear RESNET50_STOP_ITERATION if it's not itended.".format(stop_iter))
                break

        if early_stop:
            break

        mll.epoch_stop(epoch+1)
        # one epoch of training is finished
        if rank == 0:
            for name, val in eval_name_vals:
                self.logger.info('Rank[%d] Epoch[%d] Train-%s=%f', rank, epoch, name, val)
            toc = time.time()
            self.logger.info('Rank[%d] Epoch[%d] Time cost=%.3f', rank, epoch, (toc-tic))

        # sync aux params across devices
        arg_params, aux_params = self.get_params()
        self.set_params(arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data is not None and ((epoch % eval_frequency == eval_offset) or (epoch + 1 == num_epoch)):
            mll.eval_start(epoch+1, sync=True)
            res = self.score(eval_data, [validation_metric, CorrectCount(), TotalCount()],
                                score_end_callback=eval_end_callback,
                                batch_end_callback=eval_batch_end_callback, epoch=epoch)
            #TODO: pull this into default
            if rank == 0:
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            
            # temporarily add these two metrics for debugging, can be removed before submission
            res = dict(res)
            correct_count = res['correct-count']
            total_count = res['total-count']
            if 'horovod' in kvstore:
                correct_count = allreduce(correct_count)
                total_count = allreduce(total_count)
                
            acc = correct_count / total_count
            mll.eval_stop(epoch+1)
            mll.eval_accuracy(epoch+1, acc)

            mll.block_stop(block_epoch_start+1)
            if acc > accuracy_target:
                mll.run_stop(status='success')
                return

            if epoch < num_epoch - 1:
                block_epoch_start = epoch + 1
                block_epoch_count = num_epoch - epoch - 1
                if block_epoch_count > eval_frequency:
                    block_epoch_count = eval_frequency
                mll.block_start(block_epoch_start+1, count=block_epoch_count)

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()

    if profiler_on:
        mx.profiler.set_state('stop')
        print(mx.profiler.dumps())
    
    mll.run_stop(status='aborted')

def run(opt, model, train_data, val_data, lr_scheduler, context, arg_params, aux_params, logger, **kwargs):
    if opt.horovod:
        rank = hvd.rank()
        num_workers = hvd.size()
    else:
        rank = 0
        num_workers = 1

    optimizer_params = {
        'learning_rate': opt.lr,
        'wd': opt.wd,
        'momentum' : opt.momentum,
        'lr_scheduler': lr_scheduler,
        'multi_precision': False}

    if opt.horovod:
        optimizer_params['rescale_grad'] = 1. / opt.batch_size

    if opt.optimizer in {'sgdwfastlars'}:
        optimizer_params['lars'] = True
        optimizer_params['lars_eta'] = opt.lars_eta
        optimizer_params['lars_eps'] = opt.lars_eps
        mll.opt_name('sgdwfastlars')
        mll.lars_epsilon(opt.lars_eps)
        mll.lars_opt_base_learning_rate(opt.lr)
        mll.lars_opt_weight_decay(opt.wd)
        mll.lars_opt_learning_rate_warmup_epochs(opt.warmup_epochs)
        mll.lars_opt_momentum(opt.momentum)
        mll.lars_opt_end_lr(0.0001)
        mll.lars_opt_lr_decay_poly_power(2)
        mll.lars_opt_lr_decay_steps('pow2')


    if opt.horovod:
        # Setting idx2name dictionary, required to mask out entries for weight decay.
        idx2name = {}
        for i,n in enumerate(model._exec_group.param_names):
            idx2name[i] = n

        optimizer = mx.optimizer.create(opt.optimizer, sym=None, param_idx2name=idx2name, **optimizer_params)
        # Horovod: wrap optimizer with DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(optimizer)
    else:
        optimizer = mx.optimizer.create(opt.optimizer, **optimizer_params)

    # evaluation metrices
    eval_metrics = mx.metric.Accuracy()

    epoch_end_callbacks = []

    # callbacks that run after each batch
    batch_end_callbacks = []
    if opt.horovod:
        # if using horovod, only report on rank 0 with global batch size
        if rank == 0:
            batch_end_callbacks.append(mx.callback.Speedometer(
                num_workers * opt.batch_size, opt.log_interval))
    else:
        batch_end_callbacks.append(mx.callback.Speedometer(
            opt.batch_size, opt.log_interval))

    # start to train model
    fit(model,
        train_data,
        eval_data=val_data,
        eval_metric=eval_metrics,
        epoch_end_callback=epoch_end_callbacks,
        batch_end_callback=batch_end_callbacks,
        kvstore='horovod' if opt.horovod else "",
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        begin_epoch=0,
        num_epoch=opt.num_epochs,
        initializer=None,
        arg_params=arg_params,
        aux_params=aux_params,
        accuracy_target=opt.accuracy_target,
        allow_missing=True,
        eval_frequency=opt.eval_frequency,
        eval_offset=opt.eval_offset,
        logger=logger)
