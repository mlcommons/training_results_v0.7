import torch
import os

from functools import partial

from torch.utils.data import DataLoader
from mlperf_logger import log_event
from mlperf_logging.mllog import constants

from utils import COCODetection, SSDCropping, SSDTransformer
from .sampler import GeneralDistributedSampler

from pycocotools.coco import COCO

def SSDCollator(batch, is_training=False):
    # batch is: [image (300x300) Tensor, image_id, (htot, wtot), bboxes (8732, 4) Tensor, labels (8732) Tensor]
    images = []
    image_ids = []
    image_sizes = []
    bboxes = []
    bbox_offsets = [0]
    labels = []

    for item in batch:
        images.append(item[0].view(1, *item[0].shape))
        image_ids.append(item[1])
        image_sizes.append(item[2])
        bboxes.append(item[3])
        labels.append(item[4])

        bbox_offsets.append(bbox_offsets[-1] + item[3].shape[0])

    images = torch.cat(images)
    bbox_offsets = np.array(bbox_offsets).astype(np.int32)

    if is_training:
        return [images, torch.cat(bboxes), torch.cat(labels), torch.tensor(bbox_offsets)]
    else:
        return [images, torch.tensor(image_ids), image_sizes, torch.cat(bboxes), torch.cat(labels), torch.tensor(bbox_offsets)]

def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    if args.pad_input:
        mean_val.append(0.)
        std_val.append(1.)
    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    if args.nhwc:
        view = [1, 1, 1, len(mean_val)]
    else:
        view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    if args.use_fp16:
        mean = mean.half()
        std = std.half()

    return mean, std

def build_train_pipe(args):
    train_annotate = os.path.join(args.data, "annotations/bbox_only_instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    input_size = args.input_size
    train_trans = SSDTransformer((input_size, input_size), val=False)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)

    if args.distributed:
        train_sampler = GeneralDistributedSampler(train_coco, pad=False)
    else:
        train_sampler = None

    train_loader = DataLoader(train_coco,
                              batch_size=args.batch_size*args.input_batch_multiplier,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=partial(SSDCollator, is_training=True))


    return train_loader, len(train_loader)

def build_eval_pipe(args):
    # Paths
    val_annotate = os.path.join(args.data, "annotations/bbox_only_instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    input_size = args.input_size
    val_trans = SSDTransformer((input_size, input_size), val=True)
    cocoGt = COCO(annotation_file=val_annotate, use_ext=True)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans, cocoGt.dataset)
    log_event(key=constants.EVAL_SAMPLES, value=len(val_coco))

    if args.distributed:
        val_sampler = GeneralDistributedSampler(val_coco, pad=False)
    else:
        val_sampler = None

    val_dataloader   = DataLoader(val_coco,
                                  batch_size=args.eval_batch_size,
                                  shuffle=False, # Note: distributed sampler is shuffled :(
                                  sampler=val_sampler,
                                  num_workers=args.num_workers)

    inv_map = {v:k for k,v in val_coco.label_map.items()}

    return val_dataloader, inv_map, cocoGt

def build_native_pipeline(args, training=True, pipe=None):
    if training:
        return build_train_pipe(args)
    else:
        return build_eval_pipe(args)
