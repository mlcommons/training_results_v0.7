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

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring,too-many-lines
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1', 'ResNetV2',
           'BasicBlockV1', 'BasicBlockV2',
           'BottleneckV1', 'BottleneckV2',
           'ResNetMLPerf',
           'resnet18_v1', 'resnet34_v1']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from .group_batch_norm import GroupBatchNorm, GroupBatchNormAddRelu

# Helpers
def _conv3x3(channels, stride, in_channels, layout='NHWC'):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels, layout=layout,
                     cudnn_tensor_core_only=1,
                     cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1)

def _get_bn_axis_for(layout):
    return layout.find('C')

# Needed since AdaptiveAvgPooling2D is NCHW only
def _generic_AdaptiveAvgPooling2D(F, x, output_size, layout):
    if layout == 'NHWC':
        x = F.transpose(x, axes=(0, 3, 1, 2))
    x = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
    if layout == 'NHWC':
        x = F.transpose(x, axes=(0, 2, 3, 1))
    return x



class NormAddRelu(HybridBlock):
    def __init__(self, base_norm_layer, axis, bn_fp16=False, **norm_kwargs):
        super(NormAddRelu, self).__init__()
        self.norm = base_norm_layer(axis=axis,
                                    bn_fp16=bn_fp16,
                                    **({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x, residual):
        x = self.norm(x)
        x = F.Activation(x+residual, act_type='relu')
        return x


# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    fuse_bn_relu: boo. default True
        Fuse batch normalization with relu activation.
    fuse_bn_add_relu: boo. default True
        Fuse batch normalization, residual addition and relu activation.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)

        # Network body
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels, layout=layout))
        self.body.add(norm_layer(axis=_get_bn_axis_for(layout),
                                 act_type='relu' if fuse_bn_relu else None,
                                 bn_fp16=bn_fp16,
                                 **({} if norm_kwargs is None else norm_kwargs)))
        if not fuse_bn_relu:
            self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels, layout=layout))


        # Norm, addition and relu
        if fuse_bn_add_relu:
            self.bn_add_relu = GroupBatchNormAddRelu(axis=_get_bn_axis_for(layout),
                                                     bn_fp16=bn_fp16,
                                                     **({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn_add_relu = NormAddRelu(base_norm_layer=norm_layer, axis=_get_bn_axis_for(layout),
                                           bn_fp16=bn_fp16,
                                           **({} if norm_kwargs is None else norm_kwargs))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels,
                                          layout=layout,
                                          cudnn_tensor_core_only=1,
                                          cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))
            self.downsample.add(norm_layer(axis=_get_bn_axis_for(layout),
                                           bn_fp16=bn_fp16,
                                           **({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None


    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = self.bn_add_relu(x, residual)
        return x


class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride, layout=layout,
                                cudnn_tensor_core_only=1,
                                cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))
        self.body.add(norm_layer(axis=_get_bn_axis_for(layout),
                                 act_type='relu' if fuse_bn_relu else None,
                                 bn_fp16=bn_fp16,
                                 **({} if norm_kwargs is None else norm_kwargs)))
        if not fuse_bn_relu:
            self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4, layout=layout))
        self.body.add(norm_layer(axis=_get_bn_axis_for(layout),
                                 act_type='relu' if fuse_bn_relu else None,
                                 bn_fp16=bn_fp16,
                                 **({} if norm_kwargs is None else norm_kwargs)))
        if not fuse_bn_relu:
            self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1, layout=layout,
                                cudnn_tensor_core_only=1,
                                cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if not last_gamma:
            self.body.add(norm_layer(axis=_get_bn_axis_for(layout),
                                     bn_fp16=bn_fp16,
                                     **({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(axis=_get_bn_axis_for(layout), gamma_initializer='zeros',
                                     bn_fp16=bn_fp16,
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels,
                                          layout=layout,
                                          cudnn_tensor_core_only=1,
                                          cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))
            self.downsample.add(norm_layer(axis=_get_bn_axis_for(layout),
                                           bn_fp16=bn_fp16,
                                           **({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None
        self.layout = layout

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = _generic_AdaptiveAvgPooling2D(F, x, output_size=1, layout=self.layout)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.fuse_bn_relu = fuse_bn_relu

        self.bn1 = norm_layer(axis=_get_bn_axis_for(layout),
                              act_type='relu' if fuse_bn_relu else None,
                              bn_fp16=bn_fp16,
                              **({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(channels, stride, in_channels, layout=layout)
        if not last_gamma:
            self.bn2 = norm_layer(axis=_get_bn_axis_for(layout),
                                  act_type='relu' if fuse_bn_relu else None,
                                  bn_fp16=bn_fp16,
                                  **({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn2 = norm_layer(axis=_get_bn_axis_for(layout), gamma_initializer='zeros',
                                  act_type='relu' if fuse_bn_relu else None,
                                  bn_fp16=bn_fp16,
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels, 1, channels, layout=layout)

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels, layout=layout,
                                        cudnn_tensor_core_only=1,
                                        cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1)
        else:
            self.downsample = None
        self.layout = layout

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        if not self.fuse_bn_relu:
            x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        if not self.fuse_bn_relu:
            x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        if self.se:
            w = _generic_AdaptiveAvgPooling2D(F, x, output_size=1, layout=self.layout)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        return x + residual


class BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.fuse_bn_relu = fuse_bn_relu

        self.bn1 = norm_layer(axis=_get_bn_axis_for(layout),
                              act_type='relu' if fuse_bn_relu else None,
                              bn_fp16=bn_fp16,
                              **({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False,
                               layout=layout,
                               cudnn_tensor_core_only=1,
                               cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1)
        self.bn2 = norm_layer(axis=_get_bn_axis_for(layout),
                              act_type='relu' if fuse_bn_relu else None,
                              bn_fp16=bn_fp16,
                              **({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels//4, stride, channels//4, layout=layout)
        if not last_gamma:
            self.bn3 = norm_layer(axis=_get_bn_axis_for(layout),
                                  act_type='relu' if fuse_bn_relu else None,
                                  bn_fp16=bn_fp16,
                                  **({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn3 = norm_layer(axis=_get_bn_axis_for(layout), gamma_initializer='zeros',
                                  act_type='relu' if fuse_bn_relu else None,
                                  bn_fp16=bn_fp16,
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False, layout=layout,
                               cudnn_tensor_core_only=1,
                               cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1)

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels, layout=layout,
                                        cudnn_tensor_core_only=1,
                                        cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1)
        else:
            self.downsample = None
        self.layout = layout

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        if not self.fuse_bn_relu:
            F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        if not self.fuse_bn_relu:
            F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        if not self.fuse_bn_relu:
            F.Activation(x, act_type='relu')
        x = self.conv3(x)

        if self.se:
            w = _generic_AdaptiveAvgPooling2D(F, x, output_size=1, layout=self.layout)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        return x + residual


# Nets
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0, layout=layout))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, layout=layout,
                                            cudnn_tensor_core_only=1,
                                            cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))
                self.features.add(norm_layer(axis=_get_bn_axis_for(layout),
                                             act_type='relu' if fuse_bn_relu else None,
                                             bn_fp16=bn_fp16,
                                             **({} if norm_kwargs is None else norm_kwargs)))
                if not fuse_bn_relu:
                    self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1, layout=layout))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block=block, layers=num_layer, channels=channels[i+1],
                                                   stride=stride, stage_index=i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer,
                                                   fuse_bn_relu=fuse_bn_relu,
                                                   fuse_bn_add_relu=fuse_bn_add_relu,
                                                   bn_fp16=bn_fp16,
                                                   norm_kwargs=norm_kwargs,
                                                   layout=layout))
            self.features.add(nn.GlobalAvgPool2D(layout=layout))

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False,
                    norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                    norm_kwargs=None, layout='NHWC'):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer,
                            fuse_bn_relu=fuse_bn_relu,
                            fuse_bn_add_relu=fuse_bn_add_relu,
                            bn_fp16=bn_fp16,
                            norm_kwargs=norm_kwargs,
                            layout=layout))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer,
                                fuse_bn_relu=fuse_bn_relu,
                                fuse_bn_add_relu=fuse_bn_add_relu,
                                bn_fp16=bn_fp16,
                                norm_kwargs=norm_kwargs,
                                layout=layout))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


class ResNetV2(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(axis=_get_bn_axis_for(layout), scale=False, center=False,
                                         bn_fp16=bn_fp16,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0, layout=layout))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, layout=layout,
                                            cudnn_tensor_core_only=1,
                                            cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))
                self.features.add(norm_layer(axis=_get_bn_axis_for(layout),
                                             act_type='relu' if fuse_bn_relu else None,
                                             bn_fp16=bn_fp16,
                                             **({} if norm_kwargs is None else norm_kwargs)))
                if not fuse_bn_relu:
                    self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1, layout=layout))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block=block, layers=num_layer, channels=channels[i+1],
                                                   stride=stride, stage_index=i+1, in_channels=in_channels,
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer,
                                                   fuse_bn_relu=fuse_bn_relu,
                                                   fuse_bn_add_relu=fuse_bn_add_relu,
                                                   bn_fp16=bn_fp16,
                                                   norm_kwargs=norm_kwargs,
                                                   layout=layout))
                in_channels = channels[i+1]
            self.features.add(norm_layer(axis=_get_bn_axis_for(layout),
                                         act_type='relu' if fuse_bn_relu else None,
                                         bn_fp16=bn_fp16,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            if not fuse_bn_relu:
                self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D(layout=layout))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False,
                    norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                    norm_kwargs=None, layout='NHWC'):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer,
                            fuse_bn_relu=fuse_bn_relu,
                            fuse_bn_add_relu=fuse_bn_add_relu,
                            bn_fp16=bn_fp16,
                            norm_kwargs=norm_kwargs,
                            layout=layout))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer,
                                fuse_bn_relu=fuse_bn_relu,
                                fuse_bn_add_relu=fuse_bn_add_relu,
                                bn_fp16=bn_fp16,
                                norm_kwargs=norm_kwargs,
                                layout=layout))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]



class ResNetMLPerf(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NHWC'
        Dimension ordering of data and weight.
    """
    def __init__(self, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                 norm_kwargs=None, layout='NHWC', **kwargs):
        super(ResNetMLPerf, self).__init__(**kwargs)
        with self.name_scope():
            layers = [3, 4, 6] #, 3]
            channels = [64, 64, 128, 256] # , 512]
            block_type = BasicBlockV1

            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(64, 1, 0, layout=layout))
            else:
                self.features.add(nn.Conv2D(64, 7, 2, 3, use_bias=False, layout=layout,
                                            cudnn_tensor_core_only=1,
                                            cudnn_algo_fwd=1, cudnn_algo_bwd_data=1, cudnn_algo_bwd_filter=1))
                self.features.add(norm_layer(axis=_get_bn_axis_for(layout),
                                             act_type='relu' if fuse_bn_relu else None,
                                             bn_fp16=bn_fp16,
                                             **({} if norm_kwargs is None else norm_kwargs)))
                if not fuse_bn_relu:
                    self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1, layout=layout))

            self._stage_index = 1
            self.features.add(self._make_layer(block=block_type, layers=3, channels=64, stride=1, in_channels=64,
                                               last_gamma=last_gamma, use_se=use_se,
                                               norm_layer=norm_layer,
                                               fuse_bn_relu=fuse_bn_relu,
                                               fuse_bn_add_relu=fuse_bn_add_relu,
                                               bn_fp16=bn_fp16,
                                               norm_kwargs=norm_kwargs,
                                               layout=layout))
            self.features.add(self._make_layer(block=block_type, layers=4, channels=128, stride=2, in_channels=64,
                                               last_gamma=last_gamma, use_se=use_se,
                                               norm_layer=norm_layer,
                                               fuse_bn_relu=fuse_bn_relu,
                                               fuse_bn_add_relu=fuse_bn_add_relu,
                                               bn_fp16=bn_fp16,
                                               norm_kwargs=norm_kwargs,
                                               layout=layout))
            self.features.add(self._make_layer(block=block_type, layers=6, channels=256, stride=1, in_channels=128,
                                               last_gamma=last_gamma, use_se=use_se,
                                               norm_layer=norm_layer,
                                               fuse_bn_relu=fuse_bn_relu,
                                               fuse_bn_add_relu=fuse_bn_add_relu,
                                               bn_fp16=bn_fp16,
                                               norm_kwargs=norm_kwargs,
                                               layout=layout))

    def _make_layer(self, block, layers, channels, stride, in_channels=0,
                    last_gamma=False, use_se=False,
                    norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
                    norm_kwargs=None, layout='NHWC'):
        layer = nn.HybridSequential(prefix='stage%d_'%self._stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer,
                            fuse_bn_relu=fuse_bn_relu,
                            fuse_bn_add_relu=fuse_bn_add_relu,
                            bn_fp16=bn_fp16,
                            norm_kwargs=norm_kwargs,
                            layout=layout))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer,
                                fuse_bn_relu=fuse_bn_relu,
                                fuse_bn_add_relu=fuse_bn_add_relu,
                                bn_fp16=bn_fp16,
                                norm_kwargs=norm_kwargs,
                                layout=layout))
            self._stage_index += 1
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

# MLPerf
def get_mlperf_resnet(pretrained=False, layout='NHWC', ctx=cpu(), **kwargs):
    net = ResNetMLPerf(layout=layout, **kwargs)

    assert not pretrained, "pretrain code moved to top level, not supported during model build"

    return net

# Constructor
def get_resnet(version, num_layers, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`ssd.group_batch_norm.GroupBatchNorm`)
    norm_kwargs : dict
        Additional `batch_norm_layer` arguments, for example `bn_group=4`
        for :class:`ssd.group_batch_norm.GroupBatchNorm`.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        if not use_se:
            net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        else:
            net.load_parameters(get_model_file('se_resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        from gluoncv.data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net

def resnet34_v1(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_mlperf_resnet(**kwargs)
    # return get_resnet(1, 34, use_se=False, **kwargs)
