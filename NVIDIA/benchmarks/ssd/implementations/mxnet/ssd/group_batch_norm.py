import math
from mxnet.base import check_call, _LIB, c_array, _Null
from mxnet.gluon import SymbolBlock, HybridBlock
import ctypes
from mpi4py import MPI
import horovod.mxnet as hvd
import mxnet as mx
import numpy as np

USE_MPI4PY = True

anti_gc = []

def handler_bytes():
    return 64 # TODO(ahmadki): add FW function returning a real size measurement


def _init_gbn_buffers(bn_group):
    assert bn_group >= 1, 'bn_group can\'t be smaller than 1'
    if bn_group == 1:
        return _Null
    
    sync_depth = int(math.log2(bn_group))  # required sync steps
    if USE_MPI4PY:
        global_comm = MPI.COMM_WORLD
        local_comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)
        local_gpus = local_comm.Get_size()
        xbuf_ptr = (ctypes.c_void_p * local_gpus)()
        rank = hvd.local_rank()
        handler = np.zeros(handler_bytes(),dtype=np.byte)
        check_call(_LIB.MXInitXBufSingle(rank, sync_depth, xbuf_ptr, handler.ctypes.data_as(ctypes.c_void_p)))
        handlers = np.asarray([np.zeros(handler_bytes(), dtype=np.byte)]*local_gpus)
        local_comm.Allgather([handler, handler_bytes(), MPI.BYTE], [handlers, handler_bytes(), MPI.BYTE])
        check_call(_LIB.MXOpenIpcHandles(rank, local_gpus, sync_depth, xbuf_ptr, handlers.ctypes.data_as(ctypes.c_void_p)))
    else:
        local_gpus = hvd.local_size()
        xbuf_ptr = (ctypes.c_void_p * local_gpus)()
        check_call(_LIB.MXInitXBuf(local_gpus, sync_depth, xbuf_ptr))

    anti_gc.append(xbuf_ptr)
    return ctypes.addressof(xbuf_ptr)



def SymGroupBatchNorm(x, bn_group, **kwargs):
    xbuf_ptr = _init_gbn_buffers(bn_group)
    return mx.sym.BatchNorm(x, bn_group=bn_group, xbuf_ptr=xbuf_ptr, **kwargs)


class GroupBatchNorm(HybridBlock):
    """TODO(ahmadki): docstring
    Batch normalization layer (Ioffe and Szegedy, 2014) with GBN support.
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    bn_group : int, default 1
        Batch norm group size. if bn_group>1 the layer will sync mean and variance between
        all GPUs in the group. Currently only groups of 1, 2 and 4 are supported
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, bn_group=1, act_type=None, bn_fp16=False, **kwargs):
        super(GroupBatchNorm, self).__init__(**kwargs)
        self.act_type = act_type or _Null
        assert bn_group in [1, 2, 4, 8, 16]
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}
        if in_channels != 0:
            self.in_channels = in_channels
        self.bn_group = bn_group
        self.bn_fp16 = bn_fp16

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)
        self.xbuf_ptr = _init_gbn_buffers(bn_group=self.bn_group)


    def _alias(self):
        return 'batchnorm'


    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        if self.bn_fp16:
            # cast to FP32 before feeding into BN kernel
            gamma = F.cast(gamma, 'float32')
            beta = F.cast(beta, 'float32')
        return F.BatchNorm(data=x, gamma=gamma, beta=beta,
                           moving_mean=running_mean, moving_var=running_var,
                           bn_group=self.bn_group, xbuf_ptr=self.xbuf_ptr,
                           act_type=self.act_type, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ', bn_group={}'.format(self.bn_group)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))

class GroupBatchNormAddRelu(GroupBatchNorm):
    """TODO(ahmadki): docstring
    Batch normalization layer (Ioffe and Szegedy, 2014) with GBN support.
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    bn_group : int, default 1
        Batch norm group size. if bn_group>1 the layer will sync mean and variance between
        all GPUs in the group. Currently only groups of 1, 2 and 4 are supported

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def hybrid_forward(self, F, x, addend, gamma, beta, running_mean, running_var):
        if self.bn_fp16:
            # cast to FP32 before feeding into BN kernel
            gamma = F.cast(gamma, 'float32')
            beta = F.cast(beta, 'float32')
        return mx.sym.BatchNormAddRelu(data=x, addend=addend, gamma=gamma, beta=beta,
                                       moving_mean=running_mean, moving_var=running_var,
                                       bn_group=self.bn_group, xbuf_ptr=self.xbuf_ptr,
                                       name='fwd', **self._kwargs)
