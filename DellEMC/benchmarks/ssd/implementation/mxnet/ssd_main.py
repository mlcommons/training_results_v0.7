"""Train SSD"""
import os
import glob
import argparse
import logging
import random
import functools
import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet.contrib import amp
import horovod.mxnet as hvd

from mlperf_logging import mllog
from mlperf_log_utils import log_event, log_start, log_end, mpiwrapper
from mlperf_logging.mllog import constants as mlperf_constants

from input_pipeline import (get_training_iterator, get_training_pipeline,
                            get_inference_iterator, get_inference_pipeline,
                            SyntheticInputIterator)
from trainer import sgd_trainer
from loss import SSDMultiBoxLoss
from inference import COCOInference
from async_executor import AsyncExecutor
from lr_scheduler import MLPerfLearningRateScheduler
from model import SSDModel
from ssd.presets import ssd_300_resnet34_v1_mlperf_coco
from ssd.anchors import mlperf_xywh_anchors


VALID_LOGGING_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
VALID_DATA_LAYOUTS = ['NCHW', 'NHWC']
VALID_PRECISIONS = ['fp32', 'fp16', 'amp']
VALID_DATASETS = ['coco2017']  # TODO(ahmadki): add voc dataset
VALID_BACKBONES = ['resnet34_mlperf']  # TODO(ahmadki): support other backbones (JoC)
VALID_MODES = ['train', 'val', 'train_val']  # TODO(ahmadki): add inference

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')

    # Model arguments
    parser.add_argument('--mode', type=str.lower, choices=VALID_MODES, default='train_val',
                        help='Mode to run; one of %s. currently has no effect' % VALID_MODES) # TODO(ahmadki)
    parser.add_argument('--backbone', type=str, choices=VALID_BACKBONES, default='resnet34_mlperf',
                        help="Base network name which serves as feature extraction base; one of %s" % VALID_BACKBONES)
    parser.add_argument('--bn-group', type=int, default=1, choices=[1, 2, 4, 8, 16],
                        help='Group of processes to collaborate on BatchNorm ops')
    parser.add_argument('--bn-fp16', action='store_true',
                        help='Use FP16 for batchnorm gamma and beta.')
    parser.add_argument('--no-fuse-bn-add-relu', action='store_true',
                        help="Do not fuse batch norm, add and relu layers")
    parser.add_argument('--no-fuse-bn-relu', action='store_true',
                        help="Do not fuse batch norm and relu layers")
    parser.add_argument('--precision', type=str.lower, choices=VALID_PRECISIONS, default='fp16',
                        help="Data format to use; one of %s" % VALID_PRECISIONS)
    parser.add_argument('--fp16-loss-scale', type=float, default=128.0,
                        help='Static FP16 loss scale')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=VALID_DATASETS, default='coco2017',
                        help="Specify the dataset to be used; one of %s" % VALID_DATASETS)
    parser.add_argument('--synthetic', action='store_true',
                        help="Use synthetic input data")
    # TODO(ahmadki):
    # 1) Remove --use-tfrecord
    # 2) make --coco-root and --tfrecord-* mutually execlusive
    # 3) Use tfrecord by default when --tfrecord-* are used, otherwise use raw images with --coco-root
    # 4) Find a way to pass validation annotation json when using --tfrecord-* (needed for cocoapi)
    parser.add_argument('--use-tfrecord', action='store_true',
                        help="Use TFRecord instead of raw images")
    parser.add_argument('--coco-root', type=str, default='/datasets/coco2017',
                        help='Directory where coco dataset (raw images) are located.')
    parser.add_argument('--tfrecord-root', type=str, default='/datasets/coco2017/tfrecord/',
                        help='Directory where TFRecord and dali index files are located.')
    # Note: for MLPerf, --dataset-size needs to be given as an argument in order to comply with the
    # "don't touch data before run_start" rule
    parser.add_argument('--dataset-size', type=int, default=None,
                        help='Training dataset size, if none the size will be automatically inferred.')
    parser.add_argument('--eval-dataset-size', type=int, default=None,
                        help='Validation dataset size, if none the size will be automatically inferred.')
    parser.add_argument('--input-jpg-decode', type=str, default='gpu',
                        help='Way to decode jpg.')
    parser.add_argument('--hw-decoder-load', type=float, default=0.0,
                        help='Percentage of workload that will be offloaded to the hardware decoder if available. ')


    # Model data arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='The local batch size')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='The evaluation local batch size. If not specified, --batch-size will be used')
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape, use 300, 512.")  # TODO(ahmadki): support other data shapes
    parser.add_argument('--data-layout', type=str.upper, choices=VALID_DATA_LAYOUTS, default="NHWC",
                        help="Specify the input data layout; one of %s" % VALID_DATA_LAYOUTS)
    parser.add_argument('--input-batch-multiplier', type=int, default=1,
                        help="use larger batches for input pipeline")
    parser.add_argument('--dali-workers', '-j', type=int, default=6,
                        help='Number of DALI data workers, you can use larger '
                             'number to accelerate data loading, if you CPU and GPUs are powerful.')

    # General arguments
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed to be fixed.')

    # Training arguments
    parser.add_argument('--pretrained-backbone', type=str, default=None,
                        help="path to a pickle file with pretrained backbone weights. "
                             "Mutually exclusive with --resume-from")
    parser.add_argument('--epochs', type=int, default=80,
                        help='Training epochs.')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1,
                        help='Gradient predivide factor before allreduce')
    parser.add_argument('--horovod-num-groups', type=int, default=1,
                        help='num_groups argument to pass to Horovod DistributedTrainer')
    parser.add_argument('--resume-from', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./ssd_xxx_0123.params. '
                             'Mutually exclusive with --pretrained-backbone.')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='Starting epoch for resuming, default is 1 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate, if None will default to 0.0025 * (batch_size*num_gpus / 32)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='+', default=[44, 55],
                        help='epochs at which learning rate decays. default is [44, 55].')
    parser.add_argument('--lr-warmup-epochs', type=float, default=None,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--lr-warmup-factor', type=float, default=0,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='weight decay, default is 5e-4')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Save model parameters every that many epochs.')
    parser.add_argument('--async-val', action='store_true',
                        help='Execute validation asynchronously (scoring only)')
    parser.add_argument('--val-interval', type=int, default=None,
                        help='epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--val-epochs', nargs='*', type=int,
                        default=[40, 50, 55, 60, 65, 70, 75, 80],
                        help='epochs at which to evaluate in addition to --val-interval')
    parser.add_argument('--target-map', '-t', type=float, default=23,
                        help='stop training early at this threshold')
    parser.add_argument('--cocoapi-threads', type=int, default=1,
                        help='Number of OpenMP threads to use with cocoAPI')
    parser.add_argument('--nms-valid-thresh', type=float, default=0.0,
                        help='filter out results whose scores less than nms-valid-thresh.')
    parser.add_argument('--nms-overlap-thresh', type=float, default=0.5,
                        help='overlapping(IoU) threshold to suppress object with smaller score.')
    parser.add_argument('--nms-topk', type=int, default=200,
                        help='Non-Maximal Suppression (NMS) maximum number of detections')
    parser.add_argument('--post-nms', type=int, default=200,
                        help='Only return top post_nms detection results. Set to -1 to return all detections.')
    parser.add_argument('--bulk-last-wgrad', action='store_true',
                        help='Include the last wgrad in backward bulk.')

    # logging arguments
    parser.add_argument('--results', type=str, default=None,
                        help='Folder to save results. If not set, logs or weights will not be written to disk.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval.')
    parser.add_argument('--log-level', type=str, choices=VALID_LOGGING_LEVELS, default='INFO',
                        help="logging level; one of %s" % VALID_LOGGING_LEVELS)
    parser.add_argument('--log-local-ranks', type=int, nargs='+', default=[0],
                        help='use --log-level on this list of MPI ranks, '
                             'the reset will have a log level of CRITICAL')

    # profiling arguments
    parser.add_argument('--profile-no-horovod', action='store_true',
                        help='in the single gpu case, use (presumably faster) gluon.Trainer '
                             'instead of horovod.DistributedTrainer')
    parser.add_argument('--profile-start', type=int, default=None,
                        help='Iteration at which to turn on cuda profiling')
    parser.add_argument('--profile-stop', type=int, default=None,
                        help='Iteration at which to early terminate (and turn off cuda profiling)')

    # testing arguments
    parser.add_argument('--test-initialization', action='store_true',
                        help='Print network parameter statistics after initalization')
    parser.add_argument('--test-anchors', action='store_true',
                        help='Overview of normalized xywh anchors.')

    args = parser.parse_args()

    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.seed = args.seed or random.SystemRandom().randint(0, 2**31-1)
    log_event(key=mlperf_constants.SGD, value=args.seed)

    return args


def setup_logger(level='INFO', log_file=None):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(level=level)
    logger.addHandler(logging.StreamHandler())
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=level)
        logger.addHandler(fh)


def set_seed_distributed(local_seed):
    # single-element tensor with the local seed in it
    rank_0_seed = nd.full((1), local_seed, dtype=np.int32)
    if hvd.size() > 1:
        rank_0_seed = hvd.broadcast_(tensor=rank_0_seed, root_rank=0, name="broadcast_the_seed")

    nd.ndarray.waitall()
    local_seed = (rank_0_seed[0].asscalar() + hvd.rank()) % 2**31

    log_event(key=mlperf_constants.SEED, value=local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)
    mx.random.seed(local_seed)
    return local_seed


def main(async_executor=None):
    # Setup MLPerf logger
    mllog.config()
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    # Start MLPerf benchmark
    log_start(key=mlperf_constants.INIT_START, uniq=False)

    # Parse args
    args = parse_args()

    ############################################################################
    # Initialize various libraries (horovod, logger, amp ...)
    ############################################################################
    # Initialize async executor
    if args.async_val:
        assert async_executor is not None, 'Please use ssd_main_async.py to launch with async support'
    else:
        # (Force) disable async validation
        async_executor = None

    # Initialize horovod
    hvd.init()

    # Initialize AMP
    if args.precision == 'amp':
        amp.init(layout_optimization=True)

    # Set MXNET_SAFE_ACCUMULATION=1 if necessary
    if args.precision == 'fp16':
        os.environ["MXNET_SAFE_ACCUMULATION"] = "1"

    # Results folder
    network_name = f'ssd_{args.backbone}_{args.data_layout}_{args.dataset}_{args.data_shape}'
    save_prefix = None
    if args.results:
        save_prefix = os.path.join(args.results, network_name)
    else:
        logging.info("No results folder was provided. The script will not write logs or save weight to disk")

    # Initialize logger
    log_file = None
    if args.results:
        log_file = f'{save_prefix}_{args.mode}_{hvd.rank()}.log'
    setup_logger(level=args.log_level if hvd.local_rank() in args.log_local_ranks else 'CRITICAL',
                 log_file=log_file)

    # Set seed
    args.seed = set_seed_distributed(args.seed)
    ############################################################################

    ############################################################################
    # Validate arguments and print some useful information
    ############################################################################
    logging.info(args)

    assert not (args.resume_from and args.pretrained_backbone), ("--resume-from and --pretrained_backbone are "
                                                                 "mutually exclusive.")
    assert args.data_shape == 300, "only data_shape=300 is supported at the moment."
    assert args.input_batch_multiplier>=1, "input_batch_multiplier must be >= 1"
    assert not (hvd.size() == 1 and args.gradient_predivide_factor > 1), ("Gradient predivide factor is not supported "
                                                                          "with a single GPU")
    if args.data_layout == 'NCHW' or args.precision == 'fp32':
        assert args.bn_group == 1, "Group batch norm doesn't support FP32 data format or NCHW data layout."
        if not args.no_fuse_bn_relu:
            logging.warning(("WARNING: fused batch norm relu is only supported with NHWC layout. "
                             "A non fused version will be forced."))
            args.no_fuse_bn_relu = True
        if not args.no_fuse_bn_add_relu:
            logging.warning(("WARNING: fused batch norm add relu is only supported with NHWC layout. "
                             "A non fused version will be forced."))
            args.no_fuse_bn_add_relu = True
    if args.profile_no_horovod and hvd.size() > 1:
        logging.warning("WARNING: hvd.size() > 1, so must IGNORE requested --profile-no-horovod")
        args.profile_no_horovod = False

    logging.info(f'Seed: {args.seed}')
    logging.info(f'precision: {args.precision}')
    if args.precision == 'fp16':
        logging.info(f'loss scaling: {args.fp16_loss_scale}')
    logging.info(f'network name: {network_name}')
    logging.info(f'fuse bn relu: {not args.no_fuse_bn_relu}')
    logging.info(f'fuse bn add relu: {not args.no_fuse_bn_add_relu}')
    logging.info(f'bn group: {args.bn_group}')
    logging.info(f'bn all reduce fp16: {args.bn_fp16}')
    logging.info(f'MPI size: {hvd.size()}')
    logging.info(f'MPI global rank: {hvd.rank()}')
    logging.info(f'MPI local rank: {hvd.local_rank()}')
    logging.info(f'async validation: {args.async_val}')
    ############################################################################

    # TODO(ahmadki): load network and anchors based on args.backbone (JoC)
    # Load network
    net = ssd_300_resnet34_v1_mlperf_coco(pretrained_base=False,
                                          nms_overlap_thresh=args.nms_overlap_thresh,
                                          nms_topk=args.nms_topk,
                                          nms_valid_thresh=args.nms_valid_thresh,
                                          post_nms=args.post_nms,
                                          layout=args.data_layout,
                                          fuse_bn_add_relu=not args.no_fuse_bn_add_relu,
                                          fuse_bn_relu=not args.no_fuse_bn_relu,
                                          bn_fp16=args.bn_fp16,
                                          norm_kwargs={'bn_group': args.bn_group})

    # precomputed anchors
    anchors_np = mlperf_xywh_anchors(image_size=args.data_shape, clip=True, normalize=True)
    if args.test_anchors and hvd.rank() == 0:
        logging.info(f'Normalized anchors: {anchors_np}')

    # Training mode
    train_net = None
    train_pipeline = None
    trainer_fn = None
    lr_scheduler = None
    if args.mode in ['train', 'train_val']:
        # Training iterator
        num_cropping_iterations = 1
        if args.use_tfrecord:
            tfrecord_files = glob.glob(os.path.join(args.tfrecord_root, 'train.*.tfrecord'))
            index_files = glob.glob(os.path.join(args.tfrecord_root, 'train.*.idx'))
            tfrecords = [(tfrecod, index) for tfrecod, index in zip(tfrecord_files, index_files)]
        train_pipeline = get_training_pipeline(coco_root=args.coco_root if not args.use_tfrecord else None,
                                               tfrecords=tfrecords if args.use_tfrecord else None,
                                               anchors=anchors_np,
                                               num_shards=hvd.size(),
                                               shard_id=hvd.rank(),
                                               device_id=hvd.local_rank(),
                                               batch_size=args.batch_size*args.input_batch_multiplier,
                                               dataset_size=args.dataset_size,
                                               data_layout=args.data_layout,
                                               data_shape=args.data_shape,
                                               num_cropping_iterations=num_cropping_iterations,
                                               num_workers=args.dali_workers,
                                               fp16=args.precision == 'fp16',
                                               input_jpg_decode=args.input_jpg_decode,
                                               hw_decoder_load=args.hw_decoder_load,
                                               decoder_cache_size=min(100*1024*1024*1024/hvd.size(), 12*1024*1024*1024) if args.input_jpg_decode == 'cache' else 0,
                                               seed=args.seed)
        log_event(key=mlperf_constants.TRAIN_SAMPLES, value=train_pipeline.epoch_size)
        log_event(key=mlperf_constants.MAX_SAMPLES, value=num_cropping_iterations)

        # Training network
        train_net = SSDMultiBoxLoss(net=net,
                                    local_batch_size=args.batch_size,
                                    bulk_last_wgrad=args.bulk_last_wgrad)

        # Trainer function. SSDModel expects a function that takes 1 parameter - HybridBlock
        trainer_fn = functools.partial(sgd_trainer, learning_rate=args.lr,
                                       weight_decay=args.weight_decay, momentum=args.momentum,
                                       precision=args.precision, fp16_loss_scale=args.fp16_loss_scale,
                                       gradient_predivide_factor=args.gradient_predivide_factor,
                                       num_groups=args.horovod_num_groups,
                                       profile_no_horovod=args.profile_no_horovod)

        # Learning rate scheduler
        lr_scheduler = MLPerfLearningRateScheduler(learning_rate=args.lr,
                                                   decay_factor=args.lr_decay_factor,
                                                   decay_epochs=args.lr_decay_epochs,
                                                   warmup_factor=args.lr_warmup_factor,
                                                   warmup_epochs=args.lr_warmup_epochs,
                                                   epoch_size=train_pipeline.epoch_size,
                                                   global_batch_size=args.batch_size*hvd.size())

    # Validation mode
    infer_net = None
    val_iterator = None
    if args.mode in ['infer', 'val', 'train_val']:
        # Validation iterator
        tfrecord_files = glob.glob(os.path.join(args.tfrecord_root, 'val.*.tfrecord'))
        index_files = glob.glob(os.path.join(args.tfrecord_root, 'val.*.idx'))
        tfrecords = [(tfrecod, index) for tfrecod, index in zip(tfrecord_files, index_files)]
        val_pipeline = get_inference_pipeline(coco_root=args.coco_root if not args.use_tfrecord else None,
                                              tfrecords=tfrecords if args.use_tfrecord else None,
                                              num_shards=hvd.size(),
                                              shard_id=hvd.rank(),
                                              device_id=hvd.local_rank(),
                                              batch_size=args.eval_batch_size,
                                              dataset_size=args.eval_dataset_size,
                                              data_layout=args.data_layout,
                                              data_shape=args.data_shape,
                                              num_workers=args.dali_workers,
                                              fp16=args.precision == 'fp16')
        log_event(key=mlperf_constants.EVAL_SAMPLES, value=val_pipeline.epoch_size)
        
        # Inference network
        infer_net = COCOInference(net=net, ltrb=False, scale_bboxes=True, score_threshold=0.0)

        # annotations file
        cocoapi_annotation_file = os.path.join(args.coco_root, 'annotations', 'bbox_only_instances_val2017.json')

    # Prepare model
    model = SSDModel(net=net,
                     anchors_np=anchors_np,
                     precision=args.precision,
                     fp16_loss_scale=args.fp16_loss_scale,
                     train_net=train_net,
                     trainer_fn=trainer_fn,
                     lr_scheduler=lr_scheduler,
                     metric=mx.metric.Loss(),
                     infer_net=infer_net,
                     async_executor=async_executor,
                     save_prefix=save_prefix,
                     ctx=mx.gpu(hvd.local_rank()))

    # Do a training and validation runs on fake data.
    # this will set layers shape (needed before loading pre-trained backbone),
    # allocate tensors and and cache optimized graph.
    # Training dry run:
    logging.info('Running training dry runs')
    dummy_train_pipeline = get_training_pipeline(coco_root=None,
                                                 tfrecords=[('dummy.tfrecord', 'dummy.idx')],
                                                 anchors=anchors_np,
                                                 num_shards=1,
                                                 shard_id=0,
                                                 device_id=hvd.local_rank(),
                                                 batch_size=args.batch_size*args.input_batch_multiplier,
                                                 dataset_size=None,
                                                 data_layout=args.data_layout,
                                                 data_shape=args.data_shape,
                                                 num_workers=args.dali_workers,
                                                 fp16=args.precision == 'fp16',
                                                 seed=args.seed)
    dummy_train_iterator = get_training_iterator(pipeline=dummy_train_pipeline,
                                                 batch_size=args.batch_size)
    for images, box_targets, cls_targets in dummy_train_iterator:
        model.train_step(images=images, box_targets=box_targets, cls_targets=cls_targets)
    # Freeing memory is disabled due a bug in CUDA graphs
    # del dummy_train_pipeline
    # del dummy_train_iterator
    mx.ndarray.waitall()
    logging.info('Done')
    # Validation dry run:
    logging.info('Running inference dry runs')
    dummy_val_pipeline = get_inference_pipeline(coco_root=None,
                                                tfrecords=[('dummy.tfrecord', 'dummy.idx')],
                                                num_shards=1,
                                                shard_id=0,
                                                device_id=hvd.local_rank(),
                                                batch_size=args.eval_batch_size,
                                                dataset_size=None,
                                                data_layout=args.data_layout,
                                                data_shape=args.data_shape,
                                                num_workers=args.dali_workers,
                                                fp16=args.precision == 'fp16')
    dummy_val_iterator = get_inference_iterator(pipeline=dummy_val_pipeline)
    model.infer(data_iterator=dummy_val_iterator, log_interval=None)
    # Freeing memory is disabled due a bug in CUDA graphs
    # del dummy_val_pipeline
    # del dummy_val_iterator
    mx.ndarray.waitall()
    logging.info('Done')

    # re-initialize the model as a precaution in case the dry runs changed the parameters
    model.init_model(force_reinit=True)
    model.zero_grads()
    mx.ndarray.waitall()

    # load saved model or pretrained backbone
    if args.resume_from:
        model.load_parameters(filename=args.resume_from)
    elif args.pretrained_backbone:
        model.load_pretrain_backbone(picklefile_name=args.pretrained_backbone)

    # broadcast parameters
    model.broadcast_params()
    mx.ndarray.waitall()

    if args.test_initialization and hvd.rank() == 0:
        model.print_params_stats(net)

    log_end(key=mlperf_constants.INIT_STOP)

    # Main MLPerf loop (training+validation)
    mpiwrapper.barrier()
    log_start(key=mlperf_constants.RUN_START)
    mpiwrapper.barrier()
    # Real data iterators
    train_iterator = None
    val_iterator = None
    if train_pipeline:
        train_iterator = get_training_iterator(pipeline=train_pipeline,
                                               batch_size=args.batch_size,
                                               synthetic=args.synthetic)
    if val_pipeline:
        val_iterator = get_inference_iterator(pipeline=val_pipeline)
    model_map, epoch = model.train_val(train_iterator=train_iterator,
                                       start_epoch=args.start_epoch, end_epoch=args.epochs,
                                       val_iterator=val_iterator,
                                       val_interval=args.val_interval, val_epochs=args.val_epochs,
                                       annotation_file=cocoapi_annotation_file, target_map=args.target_map,
                                       train_log_interval=args.log_interval, val_log_interval=args.log_interval,
                                       save_interval=args.save_interval, cocoapi_threads=args.cocoapi_threads,
                                       profile_start=args.profile_start, profile_stop=args.profile_stop)
    status = 'success' if (model_map and model_map >= args.target_map) else 'aborted'
    mx.ndarray.waitall()
    log_end(key=mlperf_constants.RUN_STOP, metadata={"status": status})

    logging.info(f'Rank {hvd.rank()} done. map={model_map} @ epoch={epoch}')
    mx.nd.waitall()
    hvd.shutdown()


if __name__ == "__main__":
    main(async_executor=None)
