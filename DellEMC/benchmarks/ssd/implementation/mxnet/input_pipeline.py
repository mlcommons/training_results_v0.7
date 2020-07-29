"""Transforms described in https://arxiv.org/abs/1512.02325."""
import os
import json
import logging
from collections import defaultdict

import mxnet as mx
import nvidia.dali as dali
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.mxnet import DALIGenericIterator


__all__ = ['get_training_pipeline', 'get_training_iterator',
           'get_inference_pipeline', 'get_inference_iterator',
           'SyntheticInputIterator']

_mean_pixel = [255 * x for x in [0.485, 0.456, 0.406]]
_std_pixel = [255 * x for x in [0.229, 0.224, 0.225]]


def get_training_iterator(pipeline, batch_size, synthetic=False):
    iterator = DALIGenericIterator(pipelines=[pipeline],
                                   output_map=[('data', DALIGenericIterator.DATA_TAG),
                                               ('bboxes', DALIGenericIterator.LABEL_TAG),
                                               ('label', DALIGenericIterator.LABEL_TAG)],
                                   size=pipeline.shard_size,
                                   auto_reset=True)
    iterator = SSDIterator(iterator=iterator, double_buffer=True, remove_padding=False)
    if batch_size != pipeline.batch_size:
        assert not pipeline.batch_size % batch_size, "batch size must divide the pipeline batch size"
        iterator = RateMatchInputIterator(ssd_iterator=iterator,
                                          input_batch_size=pipeline.batch_size,
                                          output_batch_size=batch_size)
    if synthetic:
        iterator = SyntheticInputIterator(iterator=iterator)

    return iterator


def get_inference_iterator(pipeline):
    dali_iterator = DALIGenericIterator(pipelines=pipeline,
                                        output_map=[('data', DALIGenericIterator.DATA_TAG),
                                                    ('original_shape', DALIGenericIterator.LABEL_TAG),
                                                    ('id', DALIGenericIterator.LABEL_TAG)],
                                        size=pipeline.shard_size,
                                        auto_reset=True,
                                        squeeze_labels=False,
                                        dynamic_shape=True,
                                        fill_last_batch=False,
                                        last_batch_padded=True)
    # double buffering will not work with variable tensor shape
    iterator = SSDIterator(iterator=dali_iterator, double_buffer=False, remove_padding=True)
    return iterator


def get_training_pipeline(coco_root, tfrecords, anchors,
                          num_shards, shard_id, device_id,
                          batch_size=32, dataset_size=None,
                          data_layout="NHWC", data_shape=300,
                          num_cropping_iterations=1,
                          num_workers=6, fp16=False,
                          input_jpg_decode='gpu', hw_decoder_load=0.0, decoder_cache_size=0,
                          seed=-1):
    data_reader = get_dali_datareader(coco_root=coco_root,
                                      tfrecords=tfrecords,
                                      num_shards=num_shards,
                                      shard_id=shard_id,
                                      dataset_size=dataset_size,
                                      cache_decoder=input_jpg_decode == 'cache',
                                      training=True)
    # Training pipeline
    pipeline = SSDDALITrainPipeline(dataset_reader=data_reader,
                                    num_workers=num_workers,
                                    device_id=device_id,
                                    batch_size=batch_size,
                                    data_shape=data_shape,
                                    anchors=anchors,
                                    num_cropping_iterations=num_cropping_iterations,
                                    pad_output=data_layout == "NHWC",
                                    fp16=fp16,
                                    data_layout=data_layout,
                                    input_jpg_decode=input_jpg_decode,
                                    hw_decoder_load=hw_decoder_load,
                                    decoder_cache_size=decoder_cache_size,
                                    seed=seed)
    pipeline.build()
    return pipeline



def get_inference_pipeline(coco_root, tfrecords,
                           num_shards, shard_id, device_id,
                           batch_size=32, dataset_size=None,
                           data_layout="NHWC", data_shape=300, num_workers=6,
                           fp16=False):
    data_reader = get_dali_datareader(coco_root=coco_root,
                                      tfrecords=tfrecords,
                                      num_shards=num_shards,
                                      shard_id=shard_id,
                                      dataset_size=dataset_size,
                                      training=False)
    # Validation pipeline
    pipeline = SSDDALIValPipeline(dataset_reader=data_reader,
                                  num_workers=num_workers,
                                  device_id=device_id,
                                  batch_size=batch_size,
                                  data_shape=data_shape,
                                  pad_output=data_layout == "NHWC",
                                  fp16=fp16,
                                  data_layout=data_layout)
    pipeline.build()
    return pipeline


def get_dali_datareader(coco_root, tfrecords, num_shards, shard_id,
                        dataset_size=None, cache_decoder=False, training=True):
    '''
    tfrecords - an array of (tfrecord file, dali index file) tuples.
    '''
    assert bool(coco_root) ^ bool(tfrecords), "Exactly one of coco_root and tfrecord_root must be provided"
    if coco_root:  # Raw images reader
        logging.info("COCO reader: raw images")
        data_name = 'train2017' if training else 'val2017'
        images_root = os.path.join(coco_root, data_name)
        annotations_file = os.path.join(coco_root, 'annotations', f'bbox_only_instances_{data_name}.json')
        data_reader = DaliRawCocoDatasetReader(raw_images_root=images_root,
                                               annotations_file=annotations_file,
                                               num_shards=num_shards,
                                               shard_id=shard_id,
                                               dataset_size=dataset_size,
                                               stick_to_shard=not training or cache_decoder,
                                               pad_last_batch=not training,
                                               lazy_init=True,
                                               random_shuffle=cache_decoder,
                                               shuffle_after_epoch=training and not cache_decoder,
                                               skip_empty=training)
    else:  # TFRecord reader
        logging.info("COCO reader: TFRecord")
        data_reader = DaliTFRecordDatasetReader(tfrecord_files=[tfrecord[0] for tfrecord in tfrecords],
                                                tfrecord_index_files=[tfrecord[1] for tfrecord in tfrecords],
                                                num_shards=num_shards,
                                                shard_id=shard_id,
                                                dataset_size=dataset_size,
                                                stick_to_shard=not training or cache_decoder,
                                                pad_last_batch=not training,
                                                lazy_init=True,
                                                random_shuffle=training,
                                                skip_cached_images=cache_decoder)
    return data_reader


#################################################################
# DALI Data readers
#################################################################
class DaliTFRecordDatasetReader:
    """DALI partial pipeline with COCO Reader and loader. To be passed as
    a parameter of a DALI transform pipeline.
    Parameters
    ----------
    num_shards: int
        DALI pipeline arg - Number of pipelines used, indicating to the reader
        how to split/shard the dataset.
    shard_id: int
        DALI pipeline arg - Shard id of the pipeline must be in [0, num_shards).
    file_root
        Directory containing the COCO dataset.
    annotations_file
        The COCO annotation file to read from.
    device_id: int
        GPU device used for the DALI pipeline.
    pad_last_batch: bool
        If true, the Loader will pad the last batch with the last image
        when the batch size is not aligned with the shard size.
    """
    def __init__(self, tfrecord_files, tfrecord_index_files, num_shards, shard_id,
                 dataset_size=None, stick_to_shard=False, pad_last_batch=False,
                 initial_fill=1024, lazy_init=False, read_ahead=False, prefetch_queue_depth=1,
                 seed=-1, random_shuffle=False, skip_cached_images=False):
        self.tfrecord_files = tfrecord_files
        self.tfrecord_index_files = tfrecord_index_files
        self.num_shards = num_shards
        self.shard_id = shard_id
        features = {'image/encoded':      tfrec.FixedLenFeature([], tfrec.string, ""),
                    'image/id':           tfrec.FixedLenFeature([], tfrec.int64, -1),
                    'image/shape':        tfrec.FixedLenFeature([3], tfrec.int64, -1),
                    'image/object/label': tfrec.VarLenFeature([1], tfrec.int64, -1),
                    'image/object/bbox':  tfrec.VarLenFeature([4], tfrec.float32, 0.0),}
        # Note that (shuffle_after_epoch, skip_empty, ltrb, ratio, size_threshold) arguments
        # are not supported in the TFRecord reader, but some of them can be passed to the TFRecord
        # creation script
        self.dataset_reader = dali.ops.TFRecordReader(path=self.tfrecord_files,
                                                      index_path=self.tfrecord_index_files,
                                                      num_shards=self.num_shards,
                                                      shard_id=self.shard_id,
                                                      stick_to_shard=stick_to_shard,
                                                      pad_last_batch=pad_last_batch,
                                                      initial_fill=initial_fill,
                                                      lazy_init=lazy_init,
                                                      read_ahead=read_ahead,
                                                      prefetch_queue_depth=prefetch_queue_depth,
                                                      seed=seed,
                                                      random_shuffle=random_shuffle,
                                                      skip_cached_images=skip_cached_images,
                                                      features=features)
        self._size = dataset_size
        self.cast_int32 = dali.ops.Cast(device="cpu", dtype=dali.types.INT32)

    def __call__(self):
        """Returns three DALI graph nodes: inputs, bboxes, labels.
        To be called in `define_graph`.
        """
        inputs = self.dataset_reader()
        images = inputs["image/encoded"]
        images_shape = inputs["image/shape"]
        bboxes = inputs["image/object/bbox"]
        ids = inputs["image/id"]
        labels = self.cast_int32(inputs["image/object/label"])
        return (images, images_shape, ids, bboxes, labels)

    def __len__(self):
        if self._size is None:
            self._size = 0
            for index_file in self.tfrecord_index_files:
                with open(index_file) as fp:
                    self._size += len(fp.readlines())
        return self._size

    def epoch_size(self):
        return len(self)

    def shard_size(self, training=True):
        num_shards = self.num_shards
        shard_id = self.shard_id
        # In validation, we need the exact shard size so we don't have duplicate images. In training,
        # an approximate shard size is good enough so we have equal sized shards for better load balance.
        if not training:
            return int(len(self) * (shard_id+1) / num_shards) - int(len(self) * shard_id / num_shards)

        return len(self) // num_shards


class DaliRawCocoDatasetReader:
    """DALI partial pipeline with COCO Reader and loader. To be passed as
    a parameter of a DALI transform pipeline.
    Parameters
    ----------
    num_shards: int
        DALI pipeline arg - Number of pipelines used, indicating to the reader
        how to split/shard the dataset.
    shard_id: int
        DALI pipeline arg - Shard id of the pipeline must be in [0, num_shards).
    file_root
        Directory containing the COCO dataset.
    annotations_file
        The COCO annotation file to read from.
    device_id: int
        GPU device used for the DALI pipeline.
    shuffle_after_epoch: bool
        If true, reader shuffles whole dataset after each epoch.
    pad_last_batch: bool
        If true, the Loader will pad the last batch with the last image
        when the batch size is not aligned with the shard size.
    """
    def __init__(self, raw_images_root, annotations_file, num_shards, shard_id,
                 dataset_size=None, stick_to_shard=False, pad_last_batch=False,
                 initial_fill=1024, lazy_init=False, read_ahead=False, prefetch_queue_depth=1,
                 seed=-1, random_shuffle=False, skip_cached_images=False,
                 shuffle_after_epoch=False, skip_empty=False):
        self.raw_images_root = raw_images_root
        self.annotations_file = annotations_file
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.skip_empty = skip_empty
        self.dataset_reader = dali.ops.COCOReader(file_root=self.raw_images_root,
                                                  annotations_file=self.annotations_file,
                                                  num_shards=self.num_shards,
                                                  shard_id=self.shard_id,
                                                  stick_to_shard=stick_to_shard,
                                                  pad_last_batch=pad_last_batch,
                                                  initial_fill=initial_fill,
                                                  lazy_init=lazy_init,
                                                  read_ahead=read_ahead,
                                                  prefetch_queue_depth=prefetch_queue_depth,
                                                  seed=seed,
                                                  random_shuffle=random_shuffle,
                                                  skip_cached_images=skip_cached_images,
                                                  # dali.ops.COCOReader specific arguments
                                                  ltrb=True,
                                                  ratio=True,
                                                  save_img_ids=True,
                                                  shuffle_after_epoch=shuffle_after_epoch,
                                                  skip_empty=self.skip_empty)

        self._size = dataset_size

    def __call__(self):
        """Returns three DALI graph nodes: inputs, bboxes, labels.
        To be called in `define_graph`.
        """
        images, bboxes, labels, ids = self.dataset_reader()
        images_shape = None
        return (images, images_shape, ids, bboxes, labels)

    def __len__(self):
        if self._size is None:
            self._size = 0
            with open(self.annotations_file) as fp:
                annotations_dict = json.load(fp)
                self._size = len(annotations_dict['images'])
                if self.skip_empty:
                    images_dict = defaultdict(list)
                    for annotation in annotations_dict['annotations']:
                        image_id = str(annotation['image_id'])
                        images_dict[image_id].append(annotation)
                    # remove images with no bboxes
                    images_dict = {k: v for k, v in images_dict.items() if len(v) > 0}
                    self._size = len(images_dict.keys())
        return self._size

    def epoch_size(self):
        return len(self)

    def shard_size(self, training=True):
        num_shards = self.num_shards
        shard_id = self.shard_id
        if not training:
            return int(len(self) * (shard_id+1) / num_shards) - int(len(self) * shard_id / num_shards)

        return len(self) // num_shards

#################################################################
# DALI pipelines
#################################################################
class SSDDALITrainPipeline(dali.pipeline.Pipeline):
    """DALI Pipeline with SSD training transform.

    Parameters
    ----------
    device_id: int
         DALI pipeline arg - Device id.
    num_workers:
        DALI pipeline arg - Number of CPU workers.
    batch_size:
        Batch size.
    data_shape: int
        Height and width length. (height==width in SSD)
    anchors: list of float
        Anchors in [x,y,w,h] format normalized by image width/height.
    num_cropping_iterations: int
        Maximum number of attempts that we try to produce a valid crop
        to match a single minimum IoU value from thresholds.
    dataset_reader: float
        Partial pipeline object, which __call__ function has to return
        (images, bboxes, labels) DALI EdgeReference tuple.
    data_layout: string
        Layout of the data, represented by a string that has to be either
        "NHWC" or "NCHW".
    pad_output: bool
        Whether to pad or not the output.
    input_jpg_decode: string
        Way to decode jpg: 'gpu', 'cpu', 'cache' or 'hw'.
    hw_decoder_load: float
        Percentage of workload that will be offloaded to the hardware decoder,
        if available.
    decoder_cache_size: int
        Total size of the decoder cache in megabytes.
    """
    def __init__(self, dataset_reader, num_workers, device_id, batch_size, data_shape,
                 anchors, num_cropping_iterations, pad_output, fp16, data_layout, seed,
                 input_jpg_decode, hw_decoder_load, decoder_cache_size):
        super(SSDDALITrainPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_workers,
            seed=seed)
        # NOTATION TO MAKE IT UNDERSTANDABLE WHAT'S GOING ON:
        # prefix Dali operators as "c_", "m_", "g_" (cpu/mixed/gpu)
        # similarly operands in define_graph (cpu/gpu)
        # Dali rules:
        # 1. all kernels kwargs must be cpu operands
        # 2. cpu and mixed kernels take cpu data operands
        #    gpu kernels take gpu data operands
        # 3. cpu kernels return cpu operands
        #    mixed and gpu kernels return gpu operands

        # This is a dali.ops.COCOReader (as created, perhaps, by
        # DaliCococDatasetReader() above)
        self.c_input = dataset_reader

        # Random variables
        self.c_rng1 = dali.ops.Uniform(range=[0.5, 1.5]) # saturation and contrast
        self.c_rng2 = dali.ops.Uniform(range=[0.875, 1.125]) # brightness
        self.c_rng3 = dali.ops.Uniform(range=[-0.5, 0.5])    # hue (degrees)

        flip_probability = 0.5
        self.c_flip_coin = dali.ops.CoinFlip(probability=flip_probability) # coin_rnd

        # Augumentation techniques
        self.c_crop = dali.ops.RandomBBoxCrop(
            device="cpu",       # this operator is cpu-only
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            bbox_layout="xyXY",
            allow_no_crop=True,
            num_attempts=num_cropping_iterations)

        self.g_slice = None
        # Choose decoder
        if input_jpg_decode == 'gpu':
	    # dali does jpg decode on gpu, with ROI cropping on
            # hw decoder automatically disabled
            self.m_decode = dali.ops.ImageDecoderSlice(device="mixed", output_type=dali.types.RGB)
        elif input_jpg_decode == 'cpu':
	    # dali does jpg decode on cpu, with ROI cropping on
            # hw decoder automatically disabled
            self.m_decode = dali.ops.ImageDecoderSlice(device="cpu", output_type=dali.types.RGB)
        elif input_jpg_decode == 'cache':
	    # dali does jpg decode, with caching on and ROI cropping off
            # because the cached image needs to be reused with different crops
            # i.e. we need to unfuse decode and slice:
            self.m_decode = dali.ops.ImageDecoder(device='mixed', output_type=dali.types.RGB,
                                                  cache_size=decoder_cache_size,
                                                  cache_type="threshold",
                                                  cache_threshold=0,
                                                  hw_decoder_load=hw_decoder_load)
            self.g_slice = dali.ops.Slice(device="gpu")
        elif input_jpg_decode == 'hw':
	    # Using *ImageDecoderSlice* will automatically disable hardware accelerated decoding.
            # To make use of the hardware decoder, use *ImageDecoder* and *Slice* operators instead.
            self.m_decode = dali.ops.ImageDecoder(device='mixed', output_type=dali.types.RGB,
                                                  hw_decoder_load=hw_decoder_load)
            self.g_slice = dali.ops.Slice(device="gpu")

        # bounding-box flip

        # FIXME mfrank 2020-Feb-19 The current Dali build has a race condition
        # in the BBFlip gpu operator: see
        # https://github.com/NVIDIA/DALI/issues/1736 When
        # https://github.com/NVIDIA/DALI/pull/1738 makes it into the container
        # we should update the following to gpu (and move the corresponding
        # call in define_graph())
        self.c_bbflip = dali.ops.BbFlip(device="cpu", ltrb=True)

        self.g_resize = dali.ops.Resize(
            device="gpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        self.g_hsv = dali.ops.Hsv(device="gpu")
        self.g_brightness_contrast = dali.ops.BrightnessContrast(device="gpu")

        output_dtype = dali.types.FLOAT16 if fp16 else dali.types.FLOAT

        if data_layout == "NHWC":
            output_layout = dali.types.NHWC
        elif data_layout == "NCHW":
            output_layout = dali.types.NCHW
        else:
            raise ValueError('output_layout has to be either "NHWC" or "NCHW"')

        # operates on images
        self.g_normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=(data_shape, data_shape),
            mean=_mean_pixel,
            std=_std_pixel,
            output_dtype=output_dtype,
            output_layout=output_layout,
            pad_output=pad_output)

        # takes the list of bboxes and labels and generates (dense) tensors of
        # anchor data where each anchor is assigned a label (possibly the
        # "background" label, meaning "nothing"), and 4 bbox offsets from its
        # "default" bbox location.
        self.g_box_encoder = dali.ops.BoxEncoder(
            device="gpu",
            criteria=0.5,
            anchors=self.anchors_xywh_to_ltrb(anchors=anchors),
            offset=True,
            stds=[0.1, 0.1, 0.2, 0.2],
            scale=data_shape)

        self.g_cast = dali.ops.Cast(device="gpu", dtype=dali.types.FLOAT)

    def define_graph(self):
        """
        Define the DALI graph.
        """
        c_saturation = self.c_rng1()
        c_contrast = self.c_rng1()
        c_brightness = self.c_rng2()
        c_hue = self.c_rng3()
        c_flip = self.c_flip_coin()

        c_encoded_image, c_shape, c_ids, c_bboxes, c_labels = self.c_input()

        c_crop_begin, c_crop_size, c_bboxes, c_labels = self.c_crop(c_bboxes, c_labels)

        c_bboxes = self.c_bbflip(c_bboxes, horizontal=c_flip)

        # image manipulation
        if self.g_slice is None:
            g_images = self.m_decode(c_encoded_image, c_crop_begin, c_crop_size).gpu()
        else:
            g_images = self.m_decode(c_encoded_image).gpu()
            g_images = self.g_slice(g_images, c_crop_begin, c_crop_size)
        g_images = self.g_resize(g_images)
        g_images = self.g_hsv(
            g_images,
            saturation=c_saturation,
            hue=c_hue)
        g_images = self.g_brightness_contrast(
            g_images,
            contrast=c_contrast,
            brightness=c_brightness)
        g_images = self.g_normalize(g_images, mirror=c_flip)

        # bbox/label manipulation
        g_labels = c_labels.gpu()
        g_bboxes = c_bboxes.gpu()
        g_bboxes, g_labels = self.g_box_encoder(g_bboxes, g_labels)
        g_labels = self.g_cast(g_labels)

        return (g_images, g_bboxes, g_labels)

    def __len__(self):
        return len(self.c_input)

    @property
    def epoch_size(self):
        return self.c_input.epoch_size()

    @property
    def shard_size(self):
        return self.c_input.shard_size(training=True)

    def anchors_xywh_to_ltrb(self, anchors, x_offset=0.5, y_offset=0.5):
        anchors_ltrb = anchors.copy()
        anchors_ltrb[:, 0] = anchors[:, 0] + (x_offset - 1) * anchors[:, 2]
        anchors_ltrb[:, 1] = anchors[:, 1] + (y_offset - 1) * anchors[:, 3]
        anchors_ltrb[:, 2] = anchors[:, 0] + x_offset * anchors[:, 2]
        anchors_ltrb[:, 3] = anchors[:, 1] + y_offset * anchors[:, 3]
        return anchors_ltrb.flatten().tolist()


class SSDDALIValPipeline(dali.pipeline.Pipeline):
    """DALI Pipeline with SSD validation transform.

    Parameters
    ----------
    device_id: int
        DALI pipeline arg - Device id.
    num_workers:
        DALI pipeline arg - Number of CPU workers.
    batch_size:
        Batch size.
    data_shape: int
        Height and width length. (height==width in SSD)
    dataset_reader: float
        Partial pipeline object, which __call__ function has to return
        (images, bboxes, labels) DALI EdgeReference tuple.
    data_layout: string
        Layout of the data, represented by a string that has to be either
        "NHWC" or "NCHW".
    pad_output: bool
        Wheter to pad or not the output.
    """
    def __init__(self, num_workers, device_id, batch_size, data_shape,
                 dataset_reader, pad_output, data_layout, fp16):
        super(SSDDALIValPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_workers)

        self.c_input = dataset_reader

        self.c_resize = dali.ops.Resize(
            device="cpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        output_dtype = dali.types.FLOAT16 if fp16 else dali.types.FLOAT

        if data_layout == "NHWC":
            output_layout = dali.types.NHWC
        elif data_layout == "NCHW":
            output_layout = dali.types.NCHW
        else:
            raise ValueError('output_layout has to be either "NHWC" or "NCHW"')

        self.c_decode = dali.ops.ImageDecoder(device="cpu", output_type=dali.types.RGB)

        # mirror and normalize only
        self.g_normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            mean=_mean_pixel,
            std=_std_pixel,
            mirror=0,
            output_dtype=output_dtype,
            output_layout=output_layout,
            pad_output=pad_output)
        # have the same number of bbox per element of the batch
        self.c_shape_op = dali.ops.Shapes(device="cpu")

    def define_graph(self):
        """
        Define the DALI graph.
        """
        c_encoded_images, c_shape, c_ids, c_bboxes, c_labels = self.c_input()
        c_images = self.c_decode(c_encoded_images)
        c_shape = c_shape if c_shape is not None else self.c_shape_op(c_images)
        g_images = self.c_resize(c_images).gpu()
        g_images = self.g_normalize(g_images)
        return (g_images, c_shape.gpu(), c_ids.gpu())

    def __len__(self):
        return len(self.c_input)

    @property
    def epoch_size(self):
        return self.c_input.epoch_size()

    @property
    def shard_size(self):
        return self.c_input.shard_size(training=False)


#################################################################
# iterator wrappers
#################################################################
class SSDIterator:
    def __init__(self, iterator, double_buffer=True, remove_padding=False):
        self.iterator = iterator
        self.double_buffer = double_buffer
        self.remove_padding = remove_padding
        self._batch = None

    def __iter__(self):
        return self

    def __next__(self):
        iter_output = self.iterator.__next__()
        if isinstance(iter_output, list): # Unlike NDArrayIter, DALIGenericIterator outputs a list of ndarrays
            iter_output = iter_output[0]
        outputs = iter_output.data + iter_output.label # merge data and label lists
        padding = iter_output.pad  # number of examples padded at the end of a batch

        if self.double_buffer:
            if self._batch is None:  # create buffers on the first iteration
                self._batch = [o.copy() for o in outputs]
            else:
                for o, b in zip(outputs, self._batch):
                    o.copyto(b)
        else:
            self._batch = outputs

        if self.remove_padding and padding > 0:
            # remove invalid examples that pad the batch
            self._batch = [o[:-padding] for o in self._batch]

        return self._batch

    def reset(self):
        return None

    @property
    def batch_size(self):
        return self.iterator.batch_size

    @property
    def size(self):
        return self.iterator.size

    def epoch_size(self, pipeline_id=0):
        return self.iterator._pipes[pipeline_id].epoch_size

    def shard_size(self, pipeline_id=0):
        return self.iterator._pipes[pipeline_id].shard_size


class RateMatchInputIterator:
    def __init__(self, ssd_iterator, input_batch_size, output_batch_size):
        self.ssd_iterator = ssd_iterator
        self.input_batch_size = input_batch_size
        self.output_batch_size = output_batch_size
        self._input_buffer = None
        self._output_buffer = None
        self._offset = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self._input_buffer is None) or (self._offset >= self.input_batch_size):
            self._offset = 0
            self._input_buffer = self.ssd_iterator.__next__()
            # first iteration only:
            if self._output_buffer is None:
                self._output_buffer = [i.slice_axis(axis=0, begin=0, end=self.output_batch_size)
                                       for i in self._input_buffer]

        for o, i in zip(self._output_buffer, self._input_buffer):
            i.slice_axis(out=o, axis=0,
                         begin=self._offset,
                         end=self._offset+self.output_batch_size)

        self._offset += self.output_batch_size
        return self._output_buffer

    def reset(self):
        return None

    @property
    def batch_size(self):
        return self.output_batch_size

    @property
    def size(self):
        return self.ssd_iterator.size

    def epoch_size(self, pipeline_id=0):
        return self.ssd_iterator.epoch_size(pipeline_id=pipeline_id)

    def shard_size(self, pipeline_id=0):
        return self.ssd_iterator.shard_size(pipeline_id=pipeline_id)


class SyntheticInputIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self._buffer = None
        self._batch_size = None
        self._items_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # first iteration only:
        if self._buffer is None:
            self._buffer = [x.copy() for x in self.iterator.__next__()]
            self._batch_size = self._buffer[0].shape[0]

        if self._items_count >= self.shard_size():
            self._items_count = self._items_count % self.shard_size()
            raise StopIteration

        self._items_count += self._batch_size
        return self._buffer

    def reset(self):
        return None

    @property
    def batch_size(self):
        return self.iterator.batch_size

    @property
    def size(self):
        return self.iterator.size

    def epoch_size(self, pipeline_id=0):
        return self.iterator.epoch_size(pipeline_id=pipeline_id)

    def shard_size(self, pipeline_id=0):
        return self.iterator.shard_size(pipeline_id=pipeline_id)
