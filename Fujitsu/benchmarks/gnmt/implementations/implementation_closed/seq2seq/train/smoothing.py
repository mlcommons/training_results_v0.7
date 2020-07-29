import logging

import torch
import torch.nn as nn

from apex.contrib import xentropy
loss_func = xentropy.SoftmaxCrossEntropyLoss.apply


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, padding_idx, smoothing=0.0, fusion=True):
        """
        Constructor for the LabelSmoothing module.

        :param padding_idx: index of the PAD token
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.fusion = fusion
        logging.info(f'Fused xentropy flag set to {fusion}')

    def forward(self, x, target):
        if self.fusion:
            loss = loss_func(x, target, self.smoothing, self.padding_idx, True)
        else:
            logprobs = torch.nn.functional.log_softmax(x, dim=-1,
                                                       dtype=torch.float32)

            non_pad_mask = (target != self.padding_idx)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)[non_pad_mask]
            smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.sum()
