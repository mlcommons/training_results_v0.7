# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.optim

from . import FairseqOptimizer, register_optimizer
from apex.contrib.optimizers.fused_adam import FusedAdam
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.contrib.optimizers.distributed_fused_adam_v2 import DistributedFusedAdamV2
from apex.contrib.optimizers.distributed_fused_adam_v3 import DistributedFusedAdamV3

@register_optimizer('adam')
class FairseqAdam(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        if self.args.distributed_weight_update == 2:
            dwu_args = self.distributed_weight_update_config
            print("DistributedFusedAdam",dwu_args)
            self._optimizer = DistributedFusedAdam(params, **dwu_args, **self.optimizer_config)
        elif self.args.distributed_weight_update == 3:
            dwu_args = self.distributed_weight_update_config
            print("DistributedFusedAdamV2",dwu_args)
            self._optimizer = DistributedFusedAdamV2(params, **dwu_args, **self.optimizer_config)
        elif self.args.distributed_weight_update == 4:
            dwu_args = self.distributed_weight_update_config
            print("DistributedFusedAdamV3",dwu_args)
            self._optimizer = DistributedFusedAdamV3(params, **dwu_args, **self.optimizer_config)
        else:
            assert (self.args.distributed_weight_update == 0), "Vanilla optimizer not supported anymore"
            self._optimizer = FusedAdam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }

    @property
    def distributed_weight_update_config(self):
        """
        Return a kwarg dictionary that provides arguments for the distributed
        weight update feature.
        """
        return {
            'distributed_weight_update': self.args.distributed_weight_update,
            'dwu_group_size': self.args.dwu_group_size,
            'dwu_num_blocks': self.args.dwu_num_blocks,
            'dwu_num_chunks': self.args.dwu_num_chunks,
            'dwu_num_rs_pg': self.args.dwu_num_rs_pg,
            'dwu_num_ar_pg': self.args.dwu_num_ar_pg,
            'dwu_num_ag_pg': self.args.dwu_num_ag_pg,
            'overlap_reductions': self.args.dwu_overlap_reductions,
            'full_pipeline': self.args.dwu_full_pipeline,
            'compute_L2_grad_norm': self.args.dwu_compute_L2_grad_norm,
            'flat_mt': self.args.dwu_flat_mt,
            'e5m2_allgather': self.args.dwu_e5m2_allgather,
            'do_not_flatten_model': self.args.dwu_do_not_flatten_model,
        }

class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss
