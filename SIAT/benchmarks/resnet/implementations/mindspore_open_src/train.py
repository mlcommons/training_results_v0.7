# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


"""train_imagenet."""
import os
import argparse
import random
import numpy as np
from mind_dataset import create_dataset
from mindspore import context
from mindspore import Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.model import ParallelMode
from mindspore.train.callback import TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset.engine as de
from mindspore.communication.management import init
import mindspore.nn as nn

from metric import DistAccuracy, ClassifyCorrectCell
from mlperf_logging import mllog
from callback import StateMonitor
from model.model_thor import Model
from model.resnet import resnet50
from model.thor import THOR
import math

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
parser.add_argument('--dataset_path', type=str, default='/home/work/datasets/minddataset/train', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--image_size', type=int, default=224, help='image_size')

# eval datasets
parser.add_argument('--eval_path', type=str, default='/home/work/datasets/minddataset/eval', help='Eval dataset path')
# parser.add_argument('--eval_path', type=str, default='/cache/eval_data/', help='Eval dataset path')
parser.add_argument('--eval_batch_size', type=int, default=50, help='eval_batch_size')
parser.add_argument('--eval_image_size', type=int, default=224, help='eval_image_size')
parser.add_argument('--eval_interval', type=int, default=1, help='eval_interval')

# network
parser.add_argument('--class_num', type=int, default=1000, help='class_num')

# lr
parser.add_argument('--lr_init', type=float, default=0.14, help='learning rate init.')
parser.add_argument('--lr_decay', type=float, default=6, help='learning rate decay.')
parser.add_argument('--epoch_end', type=int, default=70, help='learning rate end epoch')
parser.add_argument('--damping_init', type=float, default=0.14, help='damping init.')
parser.add_argument('--damping_decay', type=float, default=0.85, help='damping decay rate.')
parser.add_argument('--step_per_epoch', type=int, default=625, help='step per epoch.')
parser.add_argument('--frequency', type=int, default=625, help='update frequency')
parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
parser.add_argument('--lr_warmup', type=float, default=0.01, help='lr warmup')
parser.add_argument('--epoch_size', type=int, default=50, help='train epoch size')
parser.add_argument('--lr_mode', type=str, default="poly", help='lr mode')

# optimizer
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')

# loss
parser.add_argument('--loss_scale', type=int, default=128, help='loss_scale')
parser.add_argument('--use_label_smooth', type=int, default=1, help='use_label_smooth')
parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='label_smooth_factor')

parser.add_argument('--resnet_d', type=int, default=0, help='use_resnet_d')
parser.add_argument('--init_new', type=int, default=0, help='use_new_init')

args = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
global_rank_id = int(os.getenv('RANK_ID').split("-")[-1])
local_rank_id = global_rank_id * 8 + device_id
log_filename = os.path.join(os.getcwd(), "result.log")
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
            time.sleep(1)  # delay 1 sec

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True)
context.set_context(device_id=device_id)


def get_model_lr_warmup(global_step, lr_init, decay, total_epochs, steps_per_epoch, warmup_epoch=0, lr_warmup=0.01):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epoch
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_init) - float(lr_warmup)) / float(warmup_steps)
            lr_local = float(lr_warmup) + lr_inc * (i + 1)
        else:
            epoch = (i + 1) / steps_per_epoch - warmup_epoch
            base = (1.0 - float(epoch) / total_epochs) ** decay
            lr_local = lr_init * base
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def warmup_cosine_annealing_lr(lr, max_epoch, steps_per_epoch, warmup_epochs, warmup_init_lr):
    base_lr = lr
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(base_lr) - float(warmup_init_lr)) / float(warmup_steps)
            lr = float(warmup_init_lr) + lr_inc * (i + 1)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = base_lr * decayed
        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_new(lr, max_epoch, steps_per_epoch, warmup_epochs, warmup_init_lr):
    base_lr = lr
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(base_lr) - float(warmup_init_lr)) / float(warmup_steps)
            lr = float(warmup_init_lr) + lr_inc * (i + 1)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * (i - warmup_steps) / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = base_lr * decayed
        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def get_model_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)

    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now


if __name__ == '__main__':
    context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      mirror_mean=True, parameter_broadcast=True)
    auto_parallel_context().set_all_reduce_fusion_split_indices([43], "hccl_world_groupsum1")
    auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum2")
    auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum3")
    auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum4")
    auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum5")

    # add mllog
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
    mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value="open")
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="SIAT")
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value="Ascend 910")
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value="cloud")
    mllogger.event(key=mllog.constants.CACHE_CLEAR)

    # init the distribute env
    init()

    # network
    damping = get_model_damping(0, args.damping_init, args.damping_decay, 70, args.step_per_epoch)
    net = resnet50(class_num=1000, damping=damping, loss_scale=args.loss_scale,
                   frequency=args.frequency, batch_size=args.batch_size, resnet_d=args.resnet_d,
                   init_new=args.init_new)

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)

    # loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean", smooth_factor=args.label_smooth_factor,
                                            num_classes=args.class_num)

    # train dataset
    epoch_size = args.epoch_size
    dataset = create_dataset(dataset_path=args.dataset_path, do_train=True,
                             repeat_num=epoch_size, batch_size=args.batch_size)
    step_size = dataset.get_dataset_size()
    eval_interval = args.eval_interval

    # evalutation dataset
    eval_dataset = create_dataset(dataset_path=args.eval_path, do_train=False,
                                  repeat_num=epoch_size // eval_interval, batch_size=args.eval_batch_size)
    eval_step_size = eval_dataset.get_dataset_size()

    # loss scale
    loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    if args.lr_mode == "poly":
        lr_array = get_model_lr_warmup(0, args.lr_init, args.lr_decay, args.epoch_end, args.step_per_epoch,
                                       args.warmup_epoch, args.lr_warmup)
    elif args.lr_mode == "cos":
        lr_array = warmup_cosine_annealing_lr(args.lr_init, args.epoch_end, args.step_per_epoch, args.warmup_epoch,
                                              args.lr_warmup)
    else:
        lr_array = warmup_cosine_annealing_lr_new(args.lr_init, args.epoch_end, args.step_per_epoch, args.warmup_epoch,
                                                  args.lr_warmup)

    lr = Tensor(lr_array)
    opt = THOR(filter(lambda x: x.requires_grad, net.get_parameters()), lr, args.momentum,
               filter(lambda x: 'matrix_A' in x.name, net.get_parameters()),
               filter(lambda x: 'matrix_G' in x.name, net.get_parameters()),
               filter(lambda x: 'A_inv_max' in x.name, net.get_parameters()),
               filter(lambda x: 'G_inv_max' in x.name, net.get_parameters()),
               args.weight_decay, args.loss_scale)

    # model
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2",
                  keep_batchnorm_fp32=False,
                  metrics={'acc': DistAccuracy(batch_size=args.eval_batch_size, device_num=device_num)},
                  eval_network=dist_eval_network, frequency=args.frequency)

    # set event
    mllogger.start(key=mllog.constants.INIT_START)
    mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * device_num)
    mllogger.event(key="opt_name", value="THOR")
    mllogger.event(key="base_learning_rate", value=args.lr_init)
    mllogger.event(key="learning_rate_decay_steps", value=step_size * (args.epoch_size - args.warmup_epoch))
    mllogger.event(key="learning_rate_warmup_epochs", value=args.warmup_epoch)
    mllogger.event(key="momentum", value=args.momentum)
    mllogger.event(key="weight_decay", value=args.weight_decay)

    model.init(dataset, eval_dataset)
    sync_all_rank(device_num=device_num)

    mllogger.end(key=mllog.constants.INIT_STOP)

#     # callbacks
#     state_cb = StateMonitor(data_size=step_size,
#                             tot_batch_size=args.batch_size * device_num,
#                             lrs=lr_array,
#                             device_id=device_id)
    # callbacks
    state_cb = StateMonitor(data_size=step_size,
                        mllogger=mllogger,
                        tot_batch_size=args.batch_size*device_num,
                        lrs=lr_array,
                        device_id=device_id,
                        model=model,
                        eval_dataset=eval_dataset,
                        eval_interval=eval_interval)
    
    cb = [state_cb, ]
    eval_time_cb = TimeMonitor(data_size=eval_step_size)

    # train and eval
    mllogger.start(key=mllog.constants.RUN_START)
    mllogger.event(key="train_samples", value=args.batch_size * device_num * step_size)
    mllogger.event(key="eval_samples", value=args.eval_batch_size * device_num * eval_step_size)
    
    model.train(epoch_size, dataset, callbacks=cb)
    
    mllogger.end(key=mllog.constants.RUN_STOP, metadata={"status": "success"})
    if local_rank_id == 0:
        mox.file.copy_parallel(log_filename, os.path.join(args.train_url, "result.log"))



