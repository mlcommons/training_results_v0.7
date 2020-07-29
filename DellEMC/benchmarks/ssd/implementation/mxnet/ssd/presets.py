"""SSD predefined models."""
import warnings
from .ssd import get_ssd

__all__ = ['ssd_300_resnet34_v1_mlperf_coco']

def ssd_300_resnet34_v1_mlperf_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v1 34 layers for COCO (MLPerf version).

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from gluoncv.data import COCODetection
    classes = COCODetection.CLASSES

    if kwargs.get('fuse_bn_add_relu', False):
        features = ['stage3_batchnorm12_fwd']
    else:
        features = ['stage3_normaddrelu5_activation0']

    return get_ssd('resnet34_v1', 300,
                    features=features,
                    filters=[512, 512, 256, 256, 256],
                    sizes=[21, 45, 99, 153, 207, 261, 315],
                    ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                    steps=[8, 16, 32, 64, 100, 300],
                    reduce_ratio=0.5,
                    classes=classes, dataset='coco', pretrained=pretrained,
                    pretrained_base=pretrained_base, **kwargs)
