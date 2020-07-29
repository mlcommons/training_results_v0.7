# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import torch
import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.cpu_loss = torch.empty(1, dtype=torch.float32, device=torch.device('cpu'))
        self.cpu_loss = self.cpu_loss.pin_memory()
        if args.fast_xentropy :
            from apex.contrib.xentropy import SoftmaxCrossEntropyLoss
            self.xentropy_func = SoftmaxCrossEntropyLoss.apply
        else:
            self.xentropy_func = None

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--fast-xentropy', action='store_true',
                            help='Execute fast logSoftmax and Cross Entropy function.')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        target = model.get_targets(sample, net_output).view(-1, 1)
        if self.xentropy_func is not None:
            assert (net_output[0].dtype == torch.float16) or (net_output[0].dtype == torch.float32), "Unsupported data types"
            output = net_output[0].view(net_output[0].size(0)*net_output[0].size(1),net_output[0].size(2))
            labels = target.view(target.size(0)*target.size(1))
            losses = self.xentropy_func(output, labels, self.eps, self.padding_idx, net_output[0].dtype == torch.float16)
            loss   = losses.sum()
        else :
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            non_pad_mask = target.ne(self.padding_idx)
            nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            if reduce:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # Copy the Loss to the CPU without generating a Synchronize
        self.cpu_loss.copy_(loss.detach(),non_blocking=True)
        logging_output = {
            'loss': utils.item(self.cpu_loss) if reduce else self.cpu_loss.data,
            #'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
             #'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
