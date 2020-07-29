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

import os
import argparse
import logging
import random
import signal
import math

import mxnet as mx
import numpy as np
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from fit_utils import run
from resnet50_v1b import get_symbol
from mlperf_log import mpiwrapper
import mlperf_log as mll

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLPerf ResNet50 v1.5 using MXNet')
    parser.add_argument('--use-rec', action='store_true',
                        help='use image record iter for data input. default is false.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--model', type=str, default='resnet50_v1b',
                       help='the neural network to use')
    parser.add_argument('--num-gpus', type=int, default=0,
                       help='number of gpus, 0 means using cpu')
    parser.add_argument('--num-epochs', type=int, default=72,
                       help='max num of epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-step-epochs', type=str, default="30,60",
                       help='the epochs to reduce the lr, e.g. 30,60')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    parser.add_argument('--lars-eps', type=float, default=0,
                       help='lars epsilon param')
    parser.add_argument('--lars-eta', type=float, default=0.001,
                       help='lars trust_factor param')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--eval-batch-size', type=int, default=125,
                       help='the batch size during validation')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='set the seed for python, nd and mxnet rngs')
    parser.add_argument('--batchnorm-eps', type=float, default=2e-5,
                        help='the amount added to the batchnorm variance to prevent output explosion.')
    parser.add_argument('--batchnorm-mom', type=float, default=0.9,
                        help='the leaky-integrator factor controling the batchnorm mean and variance.')
    parser.add_argument('--horovod', action="store_true", default=False,
                        help='whether to use horovod for distributed training')
    parser.add_argument('--amp', action="store_true", default=False,
                        help='whether to use BF16 model')
    parser.add_argument('--accuracy-target', type=float, default=0.759,
                        help='the MLPerf target top-1 validation accuracy')
    parser.add_argument('--eval-frequency', type=int, default=4,
                        help='evaluation every N epochs')
    parser.add_argument('--eval-offset', type=int, default=3,
                        help='first evaluation on epoch N')
    parser.add_argument('--num-examples', type=int, default=1281167,
                        help='number of images in imagenet.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    # temporarily add this option to choose model converted from
    parser.add_argument('--use-symbolic',action="store_true",
                        help='whether to use symbolic model')
    opt = parser.parse_args()
    return opt
    
def convert_from_gluon(model_name, image_shape, classes=1000, label_smoothing=0.1, last_gamma=False, ctx=mx.cpu()):
    net = get_model(name=model_name, classes=classes, pretrained=False, last_gamma=last_gamma, ctx=ctx)
    net.hybridize()
    x = mx.sym.var('data')
    y = net(x)
    y = mx.sym.SoftmaxOutput(data=y, name='softmax', smooth_alpha=label_smoothing)
    symnet = mx.symbol.load_json(y.tojson())
    return symnet

def convert_to_bf16_symbol(amp, sym, arg_params, aux_params, excluded_sym_names=[]):
    return amp.convert_model(sym,
                             arg_params,
                             aux_params,
                             target_dtype='bfloat16',
                             excluded_sym_names=excluded_sym_names,
                             cast_optional_params=False)

def get_rec_iter(opt, hvd, image_shape):
    rec_train = os.path.expanduser(opt.rec_train)
    rec_train_idx = os.path.expanduser(opt.rec_train_idx)
    rec_val = os.path.expanduser(opt.rec_val)
    rec_val_idx = os.path.expanduser(opt.rec_val_idx)

    input_size = opt.input_size
    resize = 256
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    seed = '0'
    seed_aug = 'None'
    if opt.random_seed:
        seed = opt.random_seed
        seed_aug = opt.random_seed
    
    if opt.horovod:
        num_workers = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
    else:
        num_workers = 1
        rank = 0
        local_rank = 0

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = 1,
        shuffle             = True,
        batch_size          = opt.batch_size,
        label_width         = 1,
        data_shape          = image_shape,
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        # ctx                 = 'cpu',
        num_parts           = num_workers,
        part_index          = rank,
        device_id           = local_rank,
        seed                = seed,
        seed_aug            = seed_aug,
    )

    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = 1,
        shuffle             = False,
        batch_size          = opt.batch_size,
        label_width         = 1,
        resize              = resize,
        data_shape          = image_shape,
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        # ctx                 = 'cpu',
        num_parts           = num_workers,
        part_index          = rank,
        device_id           = local_rank,
        seed                = seed,
        seed_aug            = seed_aug,
    )

    # resize train iter to ensure each machine has same number of batches per epoch
    # if not, program hangs at the end with one machine waiting for other machines
    epoch_size = math.ceil(int(opt.num_examples / num_workers) / opt.batch_size)
    train_data = mx.io.ResizeIter(train_data, epoch_size)
    return train_data, val_data

class WeightInitializer(mx.init.Xavier):
    def _init_weight(self, name, arg):
        if name.startswith("fc"):
            mx.ndarray.random.normal(0, 0.01, out=arg)
        else:
            return super()._init_weight(name, arg)

def _get_lr_scheduler(opt, num_workers):
    epoch_size = math.ceil(int(opt.num_examples / num_workers) / opt.batch_size)
    warmup_steps = epoch_size * opt.warmup_epochs
    total_steps = epoch_size * opt.num_epochs
    return mx.lr_scheduler.PolyScheduler(max_update=total_steps, base_lr=opt.lr, pwr=2, final_lr=0.0001, warmup_steps=warmup_steps)


def main():
    opt = parse_args()

    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='[%Y-%m-%d %H:%M:%S]')
    streamhandler.setFormatter(formatter)

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    logger.info(opt)
    
    mll.print_submission_info()
    mll.cache_clear()
    mll.init_start()

    if opt.horovod:
        import horovod.mxnet as hvd
        # initialize Horovod
        hvd.init()
        num_workers = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
    else:
        hvd = None
        num_workers = 1
        rank = 0
        local_rank = 0

    # Initialize seed + random number generators
    if opt.random_seed is None:
        opt.random_seed = int(random.SystemRandom().randint(0, 2**16 - 1))

    if hvd is not None:
        np.random.seed(opt.random_seed)
        all_seeds = np.random.randint(2**16, size=num_workers)
        opt.random_seed = int(all_seeds[rank])
    
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    mx.random.seed(opt.random_seed)

    #
    batch_size = opt.batch_size
    image_shape = (3, opt.input_size, opt.input_size)
    classes = 1000
    num_training_samples = 1281167 # number of images in imagenet train set
    num_val_samples = 50000
    if opt.use_rec:
        num_training_samples = len(open(opt.rec_train_idx).readlines())
        num_val_samples = len(open(opt.rec_val_idx).readlines())

    mll.seed(opt.random_seed)
    mll.training_samples(num_training_samples)
    mll.evaluation_samples(num_val_samples)
    mll.global_batch_size(batch_size * num_workers)
    mll.model_bn_span(batch_size)
    # mll.lr_rates(opt.lr)
    # mll.warmup_epoch(opt.warmup_epochs)
    mll.lr_boundary_epochs(opt.lr_step_epochs)
    mll.opt_weight_decay(opt.wd)

    if hvd is not None:
        context = mx.gpu(local_rank) if opt.num_gpus > 0 else mx.cpu(local_rank)
    else:
        context = [mx.gpu(i) for i in range(opt.num_gpus)] if opt.num_gpus > 0 else [mx.cpu()]

    if 'pow' in opt.lr_step_epochs:
        lr_scheduler = _get_lr_scheduler(opt, num_workers)
    else:
        lr_decay = opt.lr_factor
        lr_decay_period = opt.lr_decay_period
        if opt.lr_decay_period > 0:
            lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in opt.lr_step_epochs.split(',')]
        lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
        num_batches = math.ceil(num_training_samples // num_workers / batch_size)
        
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=(num_workers * opt.lr),
                        nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(opt.lr_mode, base_lr=(num_workers * opt.lr), target_lr=0,
                        nepochs=opt.num_epochs - opt.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=opt.lr_factor, power=2)
        ])

    # Weights init
    weight_initializer = WeightInitializer(rnd_type='gaussian', factor_type="in", magnitude=2)

    arg_params, aux_params = None, None
    # Create dummy data shapes and bind them to the model
    data_shapes  = [mx.io.DataDesc('data', (batch_size,) + image_shape, 'float32')]
    label_shapes = [mx.io.DataDesc('softmax_label', (batch_size,), 'float32')]

    # create model
    if not opt.use_symbolic:
        sym = convert_from_gluon(opt.model, image_shape, classes=classes, ctx=context, label_smoothing=opt.label_smoothing)
    else:
        sym = get_symbol(classes, 50, image_shape, label_smoothing=opt.label_smoothing)

    model = mx.mod.Module(context=context, symbol=sym)
    model.bind(data_shapes=data_shapes, label_shapes=label_shapes)
    model.init_params(weight_initializer, arg_params=arg_params, aux_params=aux_params)
    (arg_params, aux_params) = model.get_params()

    # AMP
    if opt.amp:
        from mxnet.contrib import amp
        # convert FP32 model to BF16
        sym, arg_params, aux_params = convert_to_bf16_symbol(amp, sym, arg_params, aux_params)
        model = mx.mod.Module(context=context, symbol=sym)
        model.bind(data_shapes=data_shapes, label_shapes=label_shapes)
        model.set_params(arg_params, aux_params)

    # Model fetch and broadcast
    if hvd is not None:
        # Horovod: fetch and broadcast parameters
        (arg_params, aux_params) = model.get_params()

        if arg_params is not None:
            hvd.broadcast_parameters(arg_params, root_rank=0)

        if aux_params is not None:
            hvd.broadcast_parameters(aux_params, root_rank=0)

        model.set_params(arg_params=arg_params, aux_params=aux_params)

    mx.ndarray.waitall()
    mll.init_stop(sync=False)
    
    # get iter of training and validation set
    train_data, val_data = get_rec_iter(opt, hvd, image_shape)

    mll.run_start()
    # start training
    run(opt, model, train_data, val_data, lr_scheduler,
        context, arg_params, aux_params, logger)

if __name__ == "__main__":
    main()
