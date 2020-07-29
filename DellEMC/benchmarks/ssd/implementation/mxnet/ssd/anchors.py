import itertools

import numpy as np
from mxnet import gluon


class SSDAnchorGenerator(gluon.HybridBlock):
    """Bounding box anchor generator for Single-shot Object Detection.

    Parameters
    ----------
    index : int
        Index of this generator in SSD models, this is required for naming.
    sizes : iterable of floats
        Sizes of anchor boxes.
    ratios : iterable of floats
        Aspect ratios of anchor boxes.
    step : int or float
        Step size of anchor boxes.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    offsets : tuple of float
        Center offsets of anchor boxes as (h, w) in range(0, 1).

    """
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        super(SSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        self._ratios = ratios
        self._index = index
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(self._index), anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
        assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                anchors.append([cx, cy, sizes[0], sizes[0]])
                anchors.append([cx, cy, sizes[1], sizes[1]])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))


class LiteAnchorGenerator(SSDAnchorGenerator):
    '''
    Bounding box anchor generator for Single-shot Object Detection, corresponding to anchors
    structure used in ssd_mobilenet_v1_coco from TF Object Detection API
    This class inherits SSDAnchorGenerator and uses the same input parameters.
    Main differences:
      - First branch is not added with another anchor with size extracted from
        the geomtric mean of current and next branch sizes.
      - First anchor in the first branch has half the size of the rest of the anchors.
      - Geometric sum anchors are added to all other branches as the last anchor.

    '''
    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
        assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                anchors.append([cx, cy, sizes[0] / 2, sizes[0] / 2])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
                if self._index > 0:
                    anchors.append([cx, cy, sizes[1], sizes[1]])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        if self._index == 0:
            return len(self._ratios)
        else:
            return len(self._sizes) + len(self._ratios) - 1


def mlperf_xywh_anchors(image_size=300, clip=True, normalize=True):
    """Generate anchors.

    Parameters
    ----------
    image_size : int
        Image width/height.
    fp16 : bool, default is False
        Whether to use fp16 precision.

    Returns
    -------
    NDArray
        Anchors in [cx, xy, w, h] format normalized by image size.

    """
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    offsets = [0.5, 0.5]

    anchors = []
    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx]
        sk2 = scales[idx+1]
        sk3 = np.sqrt(sk1*sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for alpha in aspect_ratios[idx]:
            w = sk1 * np.sqrt(alpha)
            h = sk1 / np.sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))
        # fixed location, different sizes/shapes
        for i, j in itertools.product(range(sfeat), repeat=2):
            for w, h in all_sizes:
                cx = (j + offsets[0]) * steps[idx]
                cy = (i + offsets[1]) * steps[idx]
                anchors.append((cx, cy, w, h))

    anchors = np.array(anchors)
    if clip:
        anchors = np.clip(anchors, a_min=0, a_max=image_size) # clip normalized anchors to image size
    if normalize:
        anchors /= image_size # normalize by image width/height
    return anchors
