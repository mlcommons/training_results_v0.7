"""train_imagenet."""
import os
import argparse
import random
import numpy as np
from dataset import create_dataset
from lr_generator import get_lr
from resnet import resnet50
from mindspore import context
from mindspore import Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim import LARS, Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset.engine as de
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init

from metric import DistAccuracy, ClassifyCorrectCell
from callback import StateMonitor
from mlperf_logging import mllog

import moxing as mox
import time

np.random.seed(1)
os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')
os.environ['GLOG_v'] = '3'

parser = argparse.ArgumentParser(description='Image classification')
# cloud
parser.add_argument('--data_url', type=str, default=None, help='data_url')
parser.add_argument('--train_url', type=str, default=None, help='train_url')

# train datasets
parser.add_argument('--dataset_path', type=str, default='/home/work/datasets/imagenet/train', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')

# eval datasets
parser.add_argument('--eval_path', type=str, default='/home/work/datasets/minddataset/eval', help='Eval dataset path')
parser.add_argument('--eval_batch_size', type=int, default=50, help='eval_batch_size')
parser.add_argument('--eval_interval', type=int, default=4, help='eval_interval')

# network
parser.add_argument('--class_num', type=int, default=1001, help='class_num')

# lr
parser.add_argument('--lr_decay_mode', type=str, default='poly', help='lr_decay_mode')
parser.add_argument('--poly_power', type=float, default=2, help='lars_opt_learning_rate_decay_poly_power')
parser.add_argument('--lr_init', type=float, default=0.0, help='lr_init')
parser.add_argument('--lr_max', type=float, default=25, help='lr_max')
parser.add_argument('--lr_min', type=float, default=0.0001, help='lr_min')
parser.add_argument('--max_epoch', type=int, default=68, help='max_epoch')
parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup_epochs')

# optimizer
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--use_nesterov', type=int, default=0, help='use_nesterov')
parser.add_argument('--use_lars', type=int, default=1, help='use_lars')
parser.add_argument('--lars_epsilon', type=float, default=0.0, help='lars_epsilon')
parser.add_argument('--lars_coefficient', type=float, default=0.001, help='lars_coefficient')

# loss
parser.add_argument('--loss_scale', type=int, default=1024, help='loss_scale')
parser.add_argument('--use_label_smooth', type=int, default=1, help='use_label_smooth')
parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='label_smooth_factor')

# ckpt
parser.add_argument('--save_checkpoint', type=int, default=0, help='save_checkpoint')
parser.add_argument('--save_checkpoint_epochs', type=int, default=1, help='save_checkpoint_epochs')
parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='keep_checkpoint_max')
parser.add_argument('--save_checkpoint_path', type=str, default='./outputs', help='save_checkpoint_path')

args = parser.parse_args()
args.use_nesterov = (args.use_nesterov == 1)

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
global_rank_id = int(os.getenv('RANK_ID').split("-")[-1])
local_rank_id = global_rank_id * 8 + device_id
log_filename = os.path.join(os.getcwd(), "renet50.log")
s3_rank_ready_file = os.path.join(args.train_url, 'rank_{}.txt'.format(local_rank_id))
if mox.file.exists(s3_rank_ready_file):
    mox.file.remove(s3_rank_ready_file, recursive=False)

def sync_all_rank(device_num=8):
    mox.file.write(s3_rank_ready_file, "ready")
    while local_rank_id == 0:
        all_rank_exist = True
        for rank_item in range(device_num):
            rank_fn_item = os.path.join(args.train_url, 'rank_{}.txt'.format(rank_item))
            if not mox.file.exists(rank_fn_item):
                all_rank_exist = False
        if all_rank_exist:
            break
        else:
            time.sleep(1) # delay 1 sec

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True)
context.set_context(device_id=device_id)


if __name__ == '__main__':

    context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                        mirror_mean=True, parameter_broadcast=False)
    auto_parallel_context().set_all_reduce_fusion_split_indices([85, 160])

    #mllog
    mllog.config(filename=log_filename)
    mllog.config(
        default_namespace="mindspore",
        default_stack_offset=1,
        default_clear_line=False,
        root_dir=os.path.normpath(
            os.path.dirname(os.path.realpath(__file__))))
    mllogger = mllog.get_mllogger()
    # submission
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value="resnet")
    mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value="closed")
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="SIAT")
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value="Ascend 910")
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value="cloud")
    mllogger.event(key=mllog.constants.CACHE_CLEAR)
    # init the distribute env
    init()

    # network
    net = resnet50(class_num=args.class_num)

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)

    # loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean", smooth_factor=args.label_smooth_factor, num_classes=args.class_num)

    # train dataset
    epoch_size = args.max_epoch
    dataset = create_dataset(dataset_path=args.dataset_path, do_train=True,
                             repeat_num=epoch_size, batch_size=args.batch_size)
    
    step_size = dataset.get_dataset_size()
    eval_interval = args.eval_interval
    dataset.__loop_size__ = step_size * eval_interval

    # evalutation dataset
    eval_dataset = create_dataset(dataset_path=args.eval_path, do_train=False,
                                  repeat_num=epoch_size // eval_interval, batch_size=args.eval_batch_size)
    eval_step_size = eval_dataset.get_dataset_size()

    # loss scale
    loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    # learning rate
    lr_array = get_lr(global_step=0, lr_init=args.lr_init, lr_end=args.lr_min, lr_max=args.lr_max,
                 warmup_epochs=args.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                 lr_decay_mode=args.lr_decay_mode, poly_power=args.poly_power)
    lr = Tensor(lr_array)

    # optimizer
    decayed_params = list(filter(lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name, net.trainable_params()))
    no_decayed_params = [param for param in net.trainable_params() if param not in decayed_params]
    group_params = [{'params':decayed_params, 'weight_decay':args.weight_decay},
               {'params':no_decayed_params},
               {'order_params': net.trainable_params()}]

    if args.use_lars: 
        sgd = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, args.momentum,
                       use_nesterov=args.use_nesterov)
        opt = LARS(sgd, epsilon=args.lars_epsilon, hyperpara=args.lars_coefficient, 
                   weight_decay=args.weight_decay,
                   decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name, 
                   lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name,
                   loss_scale=args.loss_scale)
    else:
        opt = Momentum(group_params, lr, args.momentum,
                   weight_decay=args.weight_decay, loss_scale=args.loss_scale, use_nesterov=args.use_nesterov)

    # model 
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2", keep_batchnorm_fp32=False, 
                  metrics={'acc':DistAccuracy(batch_size=args.eval_batch_size, device_num=device_num)},
                  eval_network=dist_eval_network)

    # set event
    mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * device_num)
    mllogger.event(key="opt_name", value="lars")
    mllogger.event(key="lars_opt_base_learning_rate", value=args.lr_max)
    mllogger.event(key="lars_opt_end_learning_rate", value=args.lr_min)
    mllogger.event(key="lars_opt_learning_rate_decay_poly_power", value=args.poly_power)
    mllogger.event(key="lars_opt_learning_rate_decay_steps", value=step_size * (epoch_size - args.warmup_epochs))
    mllogger.event(key="lars_epsilon", value=args.lars_epsilon)
    mllogger.event(key="lars_opt_learning_rate_warmup_epochs", value=args.warmup_epochs)
    mllogger.event(key="lars_opt_momentum", value=args.momentum)
    mllogger.event(key="lars_opt_weight_decay", value=args.weight_decay)

    mllogger.start(key=mllog.constants.INIT_START)
    model.init(dataset, eval_dataset)
    sync_all_rank(device_num=device_num)
    mllogger.end(key=mllog.constants.INIT_STOP)

    # callbacks
    state_cb = StateMonitor(data_size=step_size*eval_interval,
                        mllogger=mllogger,
                        tot_batch_size=args.batch_size*device_num,
                        lrs=lr_array,
                        device_id=device_id,
                        model=model,
                        eval_dataset=eval_dataset,
                        eval_interval=eval_interval)
    cb = [state_cb, ]
    if args.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_epochs*step_size,
                                    keep_checkpoint_max=args.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=args.save_checkpoint_path, config=config_ck)
        cb += [ckpt_cb]

    # train and eval
    mllogger.start(key=mllog.constants.RUN_START)
    mllogger.event(key="train_samples", value=step_size * device_num * args.batch_size)
    mllogger.event(key="eval_samples", value=eval_step_size * device_num * args.eval_batch_size)
    model.train(int(epoch_size // eval_interval), dataset, callbacks=cb)
    mllogger.event(key=mllog.constants.RUN_STOP, metadata={"status": "success"})
    if local_rank_id == 0:
        mox.file.copy_parallel(log_filename, os.path.join(args.train_url, "renet50.log"))
