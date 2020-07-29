"""Single-shot Multi-box Detector."""
import os
import logging
import warnings
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
import numpy as np
from .feature import FeatureExpander
from .predictor import ConvPredictor
from .initializer import SegmentedXavier
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from .group_batch_norm import GroupBatchNorm, GroupBatchNormAddRelu

__all__ = ['SSD', 'get_ssd']


class SSD(HybridBlock):
    """Single-shot Object Detection Network: https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    base_size : int
        Base input size, it is speficied so SSD can support dynamic input shapes.
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
        If `network` is `None`, `features` is expected to be a multi-output network.
    num_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of all categories.
    use_1x1_transition : bool
        Whether to use 1x1 convolution as transition layer between attached layers,
        it is effective reducing model capacity.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    global_pool : bool
        Whether to attach a global average pooling layer as the last output layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_overlap_thresh : float, default is 0.50.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    nms_valid_thresh: float, defualt is 0.0
        Filter input boxes to those whose scores greater than nms_valid_thresh.
    post_nms : int, default is 200
        Only return top `post_nms` detection results, You can use -1 to return all detections.
    anchor_alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `anchor_alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, scalar, etc.
    ctx : mx.Context
        Network context.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
        This will only apply to base networks that has `norm_layer` specified, will ignore if the
        base network (e.g. VGG) don't accept this argument.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NCHW'
        Dimension ordering of data and weight. Only supports 'NCHW' and 'NHWC'
        layout for now. 'N', 'C', 'H', 'W' stands for batch, channel, height,
        and width dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    predictor_kernel: tuple of int. default is (3,3)
        Dimension of predictor kernel
    predictor_pad: tuple of int. default is (1,1)
        Padding of the predictor kenrel conv.
    """
    def __init__(self, network, base_size, features, num_filters, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), anchor_alloc_size=128,
                 nms_overlap_thresh=0.5, nms_topk=200, nms_valid_thresh=0.0, post_nms=200,
                 norm_layer=GroupBatchNorm, fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False, norm_kwargs=None,
                 predictors_kernel=(3, 3), predictors_pad=(1, 1),
                 ctx=mx.cpu(), layout='NCHW', **kwargs):
        super(SSD, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        if network is None:
            num_layers = len(ratios)
        else:
            num_layers = len(features) + len(num_filters) + int(global_pool)
        assert len(sizes) == num_layers + 1
        sizes = list(zip(sizes[:-1], sizes[1:]))
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            f"Mismatched (number of layers) vs (sizes) vs (ratios): {num_layers}, {len(sizes)}, {len(ratios)}."
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.nms_overlap_thresh = nms_overlap_thresh
        self.nms_topk = nms_topk
        self.nms_valid_thresh = nms_valid_thresh
        self.post_nms = post_nms
        self.layout = layout
        self.reduce_ratio = reduce_ratio
        self._bn_fp16 = bn_fp16
        self._bn_group = norm_kwargs.get('bn_group', 1)

        logging.info(f'[SSD] network: {network}')
        logging.info(f'[SSD] norm layer: {norm_layer}')
        logging.info(f'[SSD] fuse bn relu: {fuse_bn_relu}')
        logging.info(f'[SSD] fuse bn add relu: {fuse_bn_add_relu}')
        logging.info(f'[SSD] bn group: {self._bn_group}')

        with self.name_scope():
            if network is None:
                # use fine-grained manually designed block as features
                self.features = features(pretrained=pretrained,
                                         ctx=ctx,
                                         norm_layer=norm_layer,
                                         fuse_bn_relu=fuse_bn_relu,
                                         fuse_bn_add_relu=fuse_bn_add_relu,
                                         bn_fp16=bn_fp16,
                                         norm_kwargs=norm_kwargs)
            else:
                self.features = FeatureExpander(
                    network=network, outputs=features, num_filters=num_filters,
                    use_1x1_transition=use_1x1_transition,
                    use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                    global_pool=global_pool, pretrained=pretrained, ctx=ctx,
                    norm_layer=norm_layer, fuse_bn_relu=fuse_bn_relu, fuse_bn_add_relu=fuse_bn_add_relu,
                    bn_fp16=bn_fp16,
                    norm_kwargs=norm_kwargs, layout=layout)

            # use a single ConvPredictor for conf and loc predictors (head fusion),
            # but they are treated as two different segments
            self.predictors = nn.HybridSequential()
            self.num_defaults = [4, 6, 6, 6, 4, 4]
            padding_channels_to = 8
            self.padding_amounts = []  # We keep track of padding to slice conf/loc correctly
            self.predictor_offsets = []  # We keep track of offset to initialize conf/loc correctly
            for nd in self.num_defaults:
                # keep track of beginning/ending offsets for all segments
                offsets = [0]
                n = nd * (self.num_classes + 1) # output channels for conf predictors
                offsets.append(n)
                n = n + nd * 4 # output channels for both conf and loc predictors
                offsets.append(n)
                # padding if necessary
                padding_amt = 0
                # manually pad to get HMMA kernels for NHWC layout
                if (self.layout == 'NHWC') and (n % padding_channels_to):
                    padding_amt = padding_channels_to - (n % padding_channels_to)
                    n = n + padding_amt
                    if padding_amt:
                        offsets.append(n)
                self.predictors.add(ConvPredictor(n, kernel=predictors_kernel, pad=predictors_pad, layout=layout))
                self.predictor_offsets.append(offsets)
                self.padding_amounts.append(padding_amt)

            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(self.num_classes + 1, thresh=0)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_overlap_thresh=0.5, nms_topk=200, nms_valid_thresh=0.0, post_nms=200):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_overlap_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 200
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        nms_valid_thresh: float, default is 0.0
            Filter input boxes to those whose scores greater than valid_thresh.
        post_nms : int, default is -1
            Only return top `post_nms` detection results, the rest is discarded.
            You can use -1 to return all detections.

        Returns
        -------
        None

        """
        if self.nms_overlap_thresh!=nms_overlap_thresh or \
           self.nms_topk!=nms_topk or \
           self.nms_valid_thresh!=nms_valid_thresh or \
           self.post_nms!=post_nms:
            self._clear_cached_op()
        self.nms_overlap_thresh = nms_overlap_thresh
        self.nms_topk = nms_topk
        self.nms_valid_thresh = nms_valid_thresh
        self.post_nms = post_nms

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors=None):
        """Hybrid forward"""
        features = self.features(x)
        transpose_axes = (0, 2, 3, 1) if self.layout == 'NCHW' else None
        cls_preds = []
        box_preds = []
        for feat, predictor, pad, nd in zip(features, self.predictors, self.padding_amounts, self.num_defaults):
            pred = predictor(feat)
            if transpose_axes is not None:
                pred = F.transpose(pred, transpose_axes)
            if pad: # remove padding
                pred = F.slice(pred, begin=(None, None, None, 0), end=(None, None, None, -pad))
            cls_pred = F.slice(pred, begin=(None, None, None, 0), end=(None, None, None, (self.num_classes+1) * nd))
            box_pred = F.slice(pred, begin=(None, None, None, (self.num_classes+1) * nd), end=(None, None, None, None))
            cls_pred = F.flatten(cls_pred)
            box_pred = F.flatten(box_pred)
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        if autograd.is_training():
            return [cls_preds, box_preds]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_overlap_thresh > 0 and self.nms_overlap_thresh < 1:
            result = F.contrib.box_nms(data=result,
                                       overlap_thresh=self.nms_overlap_thresh,
                                       topk=self.nms_topk,
                                       valid_thresh=self.nms_valid_thresh,
                                       id_index=0,
                                       score_index=1,
                                       coord_start=2,
                                       force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes

    # MXNet default initialization is incorrect. it for example doesn't consider data layout.
    # Furthermore, we need a custom initialization to account for head fusion and padded input
    # channel with NHWC layout
    def initialize(self, force_reinit=False, ctx=None):
        """Initialize all parameters in SSD.
           FeatureExpander:
               weight: Xavier initialization, i.e. uniform with range [-x, x] where
                   x = sqrt(6/(fan_in+fan_out)). Accept both NCHW and NHWC layouts.
               bias: uniform with range [-x, x] where x = 1/sqrt(fan_in).
           ConvPredictor:
               weight: Xavier, initialize conf/loc predictors independently.
               bias: uniform.
           ResNet:
               Use default initializer.
        """
        hw_scale = 9 # for 3x3 kernel, will be used in fan_in calculation
        in_channels = [256, 512, 512, 256, 256, 256]
        for param_name, param in self.collect_params().items():
            if "expand" in param_name: # FeatureExpander
                if "weight" in param_name:
                    param.initialize(init=SegmentedXavier(layout=self.layout),
                                     force_reinit=force_reinit, ctx=ctx)
                elif "bias" in param_name:
                    pos = param_name.find("conv") + 4 # index for layer number
                    x = int(param_name[pos]) # get layer number
                    if "trans" in param_name:
                        # first half of FeatureExpander, using 1x1 kernel 
                        # also using reduce_ratio, which affects fan_in of the second half
                        fan_in = in_channels[x]
                    else:
                        # second half of FeatureExpander, use 3x3 kernel
                        fan_in = in_channels[(x+1)] * self.reduce_ratio * hw_scale
                    scale = 1 / np.sqrt(fan_in)
                    param.initialize(init=mx.init.Uniform(scale=scale),
                                     force_reinit=force_reinit, ctx=ctx)
            elif "convpredictor" in param_name: # ConvPredictor
                pos = param_name.find("convpredictor") + 13 # index for layer number
                x = int(param_name[pos]) # get layer number
                offsets = self.predictor_offsets[x] # beginning/ending offsets for segments
                if "weight" in param_name:
                    param.initialize(init=SegmentedXavier(offsets=offsets, layout=self.layout),
                                     force_reinit=force_reinit, ctx=ctx)
                if "bias" in param_name:
                    fan_in = in_channels[x] * hw_scale
                    scale = 1 / np.sqrt(fan_in)
                    # bias is initialized with fan_in, same for all segments
                    # no need to initialize in segments, use standard Uniform
                    param.initialize(init=mx.init.Uniform(scale=scale),
                                     force_reinit=force_reinit, ctx=ctx)
            else: # ResNet
                param.initialize(force_reinit=force_reinit, ctx=ctx)

    @property
    def bn_group(self):
        return self._bn_group

    @property
    def bn_fp16(self):
        return self._bn_fp16


def get_ssd(name, base_size, features, filters, sizes, ratios, steps, classes,
            dataset, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
            fuse_bn_relu=True, fuse_bn_add_relu=True, bn_fp16=False,
            **kwargs):
    """Get SSD models.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    base_size : int
        Base image size for training, this is fixed once training is assigned.
        A fixed base size still allows you to have variable input size during test.
    features : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    bn_fp16 : bool, default False
        Whether to use FP16 for batch norm gamma and beta
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layout: str, default is 'NCHW'
        Dimension ordering of data and weight.
    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    pretrained_base = False if pretrained else pretrained_base
    base_name = None if callable(features) else name
    net = SSD(base_name, base_size, features, filters, sizes, ratios, steps,
              pretrained=pretrained_base, classes=classes, ctx=ctx,
              fuse_bn_relu=fuse_bn_relu, fuse_bn_add_relu=fuse_bn_add_relu, bn_fp16=bn_fp16,
              **kwargs)
    if pretrained:
        # TODO(ahmadki): as the file is copied from github repo, import will not work
        from ..model_store import get_model_file
        full_name = '_'.join(('ssd', str(base_size), name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net
