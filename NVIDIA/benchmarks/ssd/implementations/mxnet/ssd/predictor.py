# pylint: disable=unused-argument,arguments-differ
"""Predictor for classification/box prediction."""
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


class ConvPredictor(HybridBlock):
    """Convolutional predictor.
    Convolutional predictor is widely used in object-detection. It can be used
    to predict classification scores (1 channel per class) or box predictor,
    which is usually 4 channels per box.
    The output is of shape (N, num_channel, H, W).

    Parameters
    ----------
    num_channel : int
        Number of conv channels.
    kernel : tuple of (int, int), default (3, 3)
        Conv kernel size as (H, W).
    pad : tuple of (int, int), default (1, 1)
        Conv padding size as (H, W).
    stride : tuple of (int, int), default (1, 1)
        Conv stride size as (H, W).
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.
    in_channels : int, default is 0
        The number of input channels to this layer.
        If not specified, initialization will be deferred to the first time forward is called and
        in_channels will be inferred from the shape of input data.
    layout: str, default is 'NCHW'
        Dimension ordering of data and weight. Only supports 'NCHW' and 'NHWC'
        layout for now. 'N', 'C', 'H', 'W' stands for batch, channel, height,
        and width dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.

    """
    def __init__(self, num_channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 activation=None, use_bias=True, in_channels=0, layout='NCHW', **kwargs):
        super(ConvPredictor, self).__init__(**kwargs)
        with self.name_scope():
            # Parameters of this convolution consist of different segments that
            # need to be initialized independently. Handled in SSD.initialize()
            # with custom SegmentedXavier initializer.
            self.predictor = nn.Conv2D(
                num_channel, kernel, strides=stride, padding=pad,
                activation=activation, use_bias=use_bias, in_channels=in_channels,
                weight_initializer=None, bias_initializer=None,
                layout=layout,
                cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1)

    def hybrid_forward(self, F, x):
        return self.predictor(x)


class FCPredictor(HybridBlock):
    """Fully connected predictor.
    Fully connected predictor is used to ignore spatial information and will
    output fixed-sized predictions.


    Parameters
    ----------
    num_output : int
        Number of fully connected outputs.
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.

    """
    def __init__(self, num_output, activation=None, use_bias=True, **kwargs):
        super(FCPredictor, self).__init__(**kwargs)
        with self.name_scope():
            self.predictor = nn.Dense(
                num_output, activation=activation, use_bias=use_bias,
                weight_initializer=mx.init.Xavier(),
                bias_initializer='zeros')

    def hybrid_forward(self, F, x):
        return self.predictor(x)
