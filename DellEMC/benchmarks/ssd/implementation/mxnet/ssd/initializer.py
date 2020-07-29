"""Custom initializers for non-ResNet weights."""
import mxnet as mx
import numpy as np
from mxnet.util import is_np_array


@mx.init.register
class SegmentedXavier(mx.init.Xavier):
    """
    Returns an initializer performing "Xavier" initialization independently for
    each segment of weights.

    Parameters
    ----------
    offsets: list[int], optional
        Segment begin/end offsets, length = # segments + 1
        If None, treat the entire tensor as a single segment.

    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.

    factor_type: str, optional:
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    magnitude: float, optional
        Scale of random number.

    layout: str, default is 'NCHW'
        Dimension ordering of data and weight. Only supports 'NCHW' and 'NHWC'
        layout for now. 'N', 'C', 'H', 'W' stands for batch, channel, height,
        and width dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    """
    def __init__(self, offsets=None, rnd_type="uniform", factor_type="avg", magnitude=3, layout="NCHW"):
        super(SegmentedXavier, self).__init__(rnd_type, factor_type, magnitude)
        # check random type
        if self.rnd_type == "uniform":
            self.fn = mx.numpy.random.uniform if is_np_array() else mx.random.uniform
        elif self.rnd_type == "gaussian":
            self.fn = mx.numpy.random.normal if is_np_array() else mx.random.normal
        else:
            raise ValueError("Unknown random type")

        # check factor type
        if self.factor_type not in ["avg", "in", "out"]:
            raise ValueError("Incorrect factor type")

        self.offsets = offsets
        self.layout = layout

    def _init_weight(self, name, arr):
        shape = arr.shape
        if len(shape) <= 2:
            raise ValueError(f'This initializer cannot be applied to vector {name}.'
                             f' It assumes NCHW or NHWC layout.')
        hw_scale = 1.
        if self.layout == "NCHW":
            hw_scale = np.prod(shape[2:])
        else: # NHWC
            hw_scale = np.prod(shape[1:3])

        factor = 1.
        if self.offsets is None:
            # treat the entire tensor as a single segment
            self.offsets = [0, shape[0]]
        # initialize each weight segment independently
        for i in range(len(self.offsets)-1):
            begin, end = self.offsets[i], self.offsets[i+1]
            arr_ = arr[begin:end]
            shape_ = arr_.shape

            fan_out = shape_[0] * hw_scale
            if self.layout == 'NCHW':
                fan_in = shape_[1] * hw_scale
            else: # NHWC
                fan_in = shape_[3] * hw_scale

            factor = 1.
            if self.factor_type == "avg":
                factor = (fan_in + fan_out) / 2.0
            elif self.factor_type == "in":
                factor = fan_in
            elif self.factor_type == "out":
                factor = fan_out
            scale = np.sqrt(self.magnitude / factor)

            if self.rnd_type == "uniform":
                self.fn(-scale, scale, shape_, out=arr_)
            elif self.rnd_type == "gaussian":
                self.fn(0, scale, shape_, out=arr_)
