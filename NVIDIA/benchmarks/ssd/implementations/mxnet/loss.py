import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _reshape_like
import horovod.mxnet as hvd


def _as_list(arr):
    """Make sure input is a list of mxnet NDArray"""
    if not isinstance(arr, (list, tuple)):
        return [arr]
    return arr


class SSDMultiBoxLoss(Loss):
    r"""Single-Shot Multibox Object Detection Loss.
    .. note::
        Since cross device synchronization is required to compute batch-wise statistics,
        it is slightly sub-optimal compared with non-sync version. However, we find this
        is better for converged model performance.
    Parameters
    ----------
    local_batch_size: int
        The size of mini-batch.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.
    negative_mining_ratio : float, default is 3
        Ratio of negative vs. positive samples.
    rho : float, default is 1.0
        Threshold for trimmed mean estimator. This is the smooth parameter for the
        L1-L2 transition.
    lambd : float, default is 1.0
        Relative weight between classification and box regression loss.
        The overall loss is computed as :math:`L = loss_{class} + \lambda \times loss_{loc}`.
    min_hard_negatives : int, default is 0
        Minimum number of negatives samples.
    """
    def __init__(self, net, local_batch_size, bulk_last_wgrad=False,
                 batch_axis=0, weight=None, negative_mining_ratio=3,
                 rho=1.0, lambd=1.0, min_hard_negatives=0, **kwargs):
        super(SSDMultiBoxLoss, self).__init__(weight, batch_axis, **kwargs)
        self.net = net
        self.bulk_last_wgrad = bulk_last_wgrad
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd
        self._min_hard_negatives = max(0, min_hard_negatives)
        # precomute arange functions for scatter_nd
        self.s0 = local_batch_size
        self.s1 = 8732  # TODO(ahmadki): hard coded :(
        r_init = mx.nd.arange(0, self.s1).tile(reps=(self.s0, 1)).reshape((1,-1)).squeeze(axis=0)
        idx_r_init = mx.nd.arange(0, self.s0).repeat(repeats=self.s1) # row indices
        with self.name_scope():
            self.cls_target = self.params.get('cls_target',
                                              shape=(local_batch_size, self.s1),
                                              differentiable=False)
            self.box_target = self.params.get('box_target',
                                              shape=(local_batch_size, self.s1, 4),
                                              differentiable=False)
            self.r = self.params.get('r', shape=r_init.shape,
                                     init=mx.initializer.Constant(r_init),
                                     differentiable=False)
            self.idx_r = self.params.get('idx_r', shape=idx_r_init.shape,
                                         init=mx.initializer.Constant(idx_r_init),
                                         differentiable=False)

    def hybrid_forward(self, F, images, cls_target, box_target, r, idx_r):
        if self.bulk_last_wgrad:
            # make the last wgrad use the copy of the input
            # so it joins the bulk
            images = F.identity(images)
        cls_pred, box_pred = self.net(images)

        # loss needs to be done in FP32
        cls_pred = cls_pred.astype(dtype='float32')
        box_pred = box_pred.astype(dtype='float32')

        pred = F.log_softmax(cls_pred, axis=-1)
        pos = cls_target > 0
        pos_num = pos.sum(axis=1)

        cls_loss = -F.pick(pred, cls_target, axis=-1, keepdims=False)
        idx = (cls_loss * (pos - 1)).argsort(axis=1)
        # use scatter_nd to save one argsort
        idx_c = idx.reshape((1,-1)).squeeze(axis=0) # column indices
        idx = F.stack(idx_r, idx_c)
        rank = F.scatter_nd(r, idx, (self.s0, self.s1))
        hard_negative = F.broadcast_lesser(rank, F.maximum(self._min_hard_negatives, pos.sum(axis=1) * self._negative_mining_ratio).expand_dims(-1))
        # mask out if not positive or negative
        cls_loss = F.where((pos + hard_negative) > 0, cls_loss, F.zeros_like(cls_loss))
        cls_loss = F.sum(cls_loss, axis=0, exclude=True)

        box_pred = _reshape_like(F, box_pred, box_target)
        box_loss = F.abs(box_pred - box_target)
        box_loss = F.smooth_l1(data=box_loss, scalar=1.0)
        # box loss only apply to positive samples
        box_loss = F.broadcast_mul(box_loss, pos.expand_dims(axis=-1))
        box_loss = F.sum(box_loss, axis=0, exclude=True)

        # normalize loss with num_pos_per_image
        # see https://github.com/mlperf/training/blob/master/single_stage_detector/ssd/base_model.py#L201-L204
        num_mask = (pos_num > 0).astype('float32')
        pos_num = pos_num.astype('float32').clip(a_min=1e-6, a_max=8732)
        sum_loss = (num_mask * (cls_loss + self._lambd * box_loss) / pos_num).mean(axis=0)

        return sum_loss
