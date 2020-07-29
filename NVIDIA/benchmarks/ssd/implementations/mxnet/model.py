"""SSD Model"""
import tempfile
import os
import logging
import warnings
import time
import math

import numpy as np
import horovod.mxnet as hvd
from mpi4py import MPI
from mxnet import cuda_utils as cu
import mxnet as mx
from mxnet import autograd
from mxnet.contrib import amp

from mlperf_logging.mllog import constants as mlperf_constants
from mlperf_log_utils import log_event, log_start, log_end

from ssd.pretrain import pretrain_backbone
from coco import coco_map_score

comm = MPI.COMM_WORLD

class SSDModel:
    FP16_BLACKLIST = ['batchnorm']  # List of layers that can't be converted to FP16
    RUNNING_PARAMS = ['running_mean', 'running_var']  # List of running params that need to be allreduced across ranks
    # TODO(ahmadki): there is a bug with horovod that breaks allgather for mxnet.
    # As a temporary workaround we use mpi4py
    ALL_GATHER_WITH_MPI4PY = True

    def __init__(self, net, anchors_np, precision='fp16', fp16_loss_scale=128,
                 train_net=None, trainer_fn=None, lr_scheduler=None, metric=None,
                 infer_net=None, async_executor=None, save_prefix=None, ctx=None):
        self.save_prefix = save_prefix
        self.ctx = ctx or mx.gpu(hvd.local_rank())

        self.net = net
        self.anchors_mx = mx.nd.array(anchors_np,
                                      dtype='float16' if precision == 'fp16' else 'float32',
                                      ctx=self.ctx)
        self.precision = precision
        self.fp16_loss_scale = fp16_loss_scale

        # Train params
        self.train_net = train_net
        self.lr_scheduler = lr_scheduler
        self.metric = metric

        # Validation params
        self.infer_net = infer_net
        self.async_executor = async_executor

        self.init_model()
        self.trainer = trainer_fn(self.net) if trainer_fn else None

    def init_model(self, force_reinit=False, ctx=None):
        self.ctx = ctx or self.ctx

        # FP16 mode
        if self.precision == 'fp16':
            # Convert anchors to FP16
            self.anchors_mx = self.anchors_mx.astype('float16')
            # Convert network to FP16
            for param_name, param in self.net.collect_params().items():
                if self.net.bn_fp16:
                    # for BN, only cast gamma and beta to FP16
                    # running_mean and running_var stay in FP32
                    if not any(e in param_name for e in self.RUNNING_PARAMS):
                        param.cast('float16')
                else:
                    # do not cast BN parameters
                    if not any(blst in param_name for blst in self.FP16_BLACKLIST):
                        param.cast('float16')

        # Initialize and hybridize networks
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") 
            # network+loss
            if self.train_net:
                self.train_net.initialize(force_reinit=force_reinit, ctx=self.ctx)
                self.train_net.hybridize(static_alloc=True, static_shape=True)
            # network+cocoapi
            if self.infer_net:
                self.infer_net.initialize(force_reinit=force_reinit, ctx=self.ctx)
                self.infer_net.hybridize(static_alloc=True, static_shape=True)

            # SSD network has a custom initializer (see notes at ssd.py), therefore we initialize it
            # at the end to overwrite train_net and infer_net initialization
            self.net.initialize(force_reinit=force_reinit, ctx=self.ctx)
            self.net.hybridize(static_alloc=True, static_shape=True)

    def train_val(self, train_iterator, start_epoch=1, end_epoch=80,
                  val_iterator=None, val_interval=None, val_epochs=None,
                  annotation_file=None, target_map=0.23,
                  train_log_interval=100, val_log_interval=100,
                  save_interval=None, cocoapi_threads=1,
                  profile_start=None, profile_stop=None):
        local_train_batch_size = train_iterator.batch_size
        global_train_batch_size = local_train_batch_size * hvd.size()
        log_event(key=mlperf_constants.MODEL_BN_SPAN, value=self.bn_group*local_train_batch_size)
        log_event(key=mlperf_constants.GLOBAL_BATCH_SIZE, value=global_train_batch_size)
        epoch_size = train_iterator.epoch_size()
        iterations_per_epoch = math.ceil(epoch_size / global_train_batch_size)

        logging.info(f'Training from epoch: {start_epoch}')
        for epoch in range(start_epoch, end_epoch+1):
            # Train for 1 epoch
            ret = self.train_epoch(data_iterator=train_iterator,
                                   global_train_batch_size=global_train_batch_size,
                                   iterations_per_epoch=iterations_per_epoch,
                                   epoch=epoch, log_interval=train_log_interval,
                                   profile_start=profile_start, profile_stop=profile_stop)
            if ret > 0:
                return None, epoch

            val_map = None
            val_epoch = epoch
            # Run (or schedule) a validation run
            if (val_interval and not epoch % val_interval) or (val_epochs and epoch in val_epochs):
                self.allreduce_running()  # all reduce the running parameters
                val_map = self.validate(val_iterator=val_iterator,
                                        epoch=epoch,
                                        annotation_file=annotation_file,
                                        cocoapi_threads=cocoapi_threads,
                                        log_interval=val_log_interval)

            # Check if there are completed async validation runs
            if self.async_executor:
                val_epoch, val_map = self.get_async_results(waitall=epoch == end_epoch)

            # Check if target accuracy reached
            if val_map and val_map >= target_map:
                if self.save_prefix and hvd.rank() == 0:
                    save_fname = f'{self.save_prefix}_epoch{epoch}_map{val_map:.2f}.params'
                    logging.info(f'Saving model weights: {save_fname}')
                    if self.async_executor:
                        src_fname = os.path.join(tempfile.gettempdir(), f'temp_ssd_mxnet_epoch{val_epoch}.params')
                        os.rename(src_fname, save_fname)
                    else:
                        self.net.save_parameters(save_fname)
                return val_map, val_epoch

            # Save model weights
            if (save_interval and not epoch % save_interval) and hvd.rank() == 0 and self.save_prefix:
                save_fname = f'{self.save_prefix}_epoch{epoch}.params'
                logging.info(f'Saving model weights: {save_fname}')
                self.net.save_parameters(save_fname)

        return None, epoch

    def train_epoch(self, data_iterator, global_train_batch_size, iterations_per_epoch,
                    epoch=1, log_interval=None, profile_start=None, profile_stop=None):
        current_iter = (epoch - 1) * iterations_per_epoch + 1
        timing_iter_count = 0
        timing_iter_last_tick = time.time()
        epoch_start_time = time.time()
        log_start(key=mlperf_constants.EPOCH_START,
                  metadata={'epoch_num': epoch, 'current_iter_num': 0}) # FIXME(mfrank)
        self.metric.reset()  # Reset epoch metrics
        for i, (images, box_targets, cls_targets) in enumerate(data_iterator):
            if profile_start is not None and current_iter == profile_start:
                cu.cuda_profiler_start()

            if profile_stop is not None and current_iter >= profile_stop:
                if profile_start is not None and current_iter >= profile_start:
                    # we turned cuda profiling on, better turn it off too
                    cu.cuda_profiler_stop()
                return 1

            self.metric.reset_local() # Reset iter metrics
            lr = self.lr_scheduler(current_epoch=epoch,
                                   current_iter=current_iter)
            self.trainer.set_learning_rate(lr)  # Set Learning rate
            sum_loss = self.train_step(images=images, cls_targets=cls_targets, box_targets=box_targets)
            self.trainer.step(1)
            sum_loss = sum_loss.as_in_context(mx.cpu())
            self.metric.update(0, sum_loss) # Update metric

            timing_iter_count = timing_iter_count + 1
            if log_interval and not current_iter % log_interval:
                name0, loss0 = self.metric.get()
                mx.nd.waitall()
                timing_tick = time.time()
                iter_time = timing_tick - timing_iter_last_tick
                iteration_prefix = (f'[Training][Iteration {current_iter}][Epoch {epoch}, '
                                    f'Batch {i+1}/{iterations_per_epoch}]')
                if hvd.rank() == 0:
                    logging.info((f'{iteration_prefix} '
                                  f'lr: {lr:.5f}, '
                                  f'training time: {iter_time*1000.0/timing_iter_count:.3f} [ms], '
                                  f'speed: {global_train_batch_size*timing_iter_count/iter_time:.3f} [imgs/sec], '
                                  f'{name0}={loss0:.3f}'))
                # TODO(ahmadki): remove once NaN issues are solved
                if np.isnan(loss0):
                    logging.info(f'{iteration_prefix} NaN detected in rank {hvd.rank()}. terminating.')
                    return 2
                timing_iter_count = 0
                timing_iter_last_tick = timing_tick
            current_iter = current_iter + 1

        name0, loss0 = self.metric.get_global()
        mx.nd.waitall() # cpu has been launching async
        epoch_time = time.time()-epoch_start_time
        if log_interval and hvd.rank() == 0:
            logging.info((f'[Training][Epoch {epoch}] '
                          f'training time: {epoch_time:.3f} [sec],'
                          f'avg speed: {(i+1)*global_train_batch_size/epoch_time:.3f} [imgs/sec],'
                          f'{name0}={loss0:.3f}'))

        log_end(key=mlperf_constants.EPOCH_STOP, metadata={'epoch_num': epoch})
        return 0

    def train_step(self, images, cls_targets, box_targets):
        # treat targets as parameters of loss
        # so they are internal tensor of the network
        # no problem of bulks accessing external inputs
        # which makes loss completely sync free
        # copy box_targets and cls_targets to parameters
        for param_name, param in self.train_net.collect_params().items():
            if "box_target" in param_name:
                param.set_data(box_targets)
            elif "cls_target" in param_name:
                param.set_data(cls_targets)

        with autograd.record():
            sum_loss = self.train_net(images)
            if self.precision == 'amp':
                with amp.scale_loss(sum_loss, self.trainer) as scaled_loss:
                    autograd.backward(scaled_loss)
            elif self.precision == 'fp16':
                scaled_loss = sum_loss * self.fp16_loss_scale
                autograd.backward(scaled_loss)
            else:
                autograd.backward(sum_loss)

        return sum_loss


    def validate(self, val_iterator, epoch=1, annotation_file=None, cocoapi_threads=1, log_interval=None):
        """Test on validation dataset."""
        log_start(key=mlperf_constants.EVAL_START,
                  metadata={'epoch_num': epoch})
        time_ticks = [time.time()]
        time_messages = []

        # save a copy of weights to temp dir
        if self.async_executor and self.save_prefix and hvd.rank() == 0:
            save_fname = os.path.join(tempfile.gettempdir(), f'temp_ssd_mxnet_epoch{epoch}.params')
            self.net.save_parameters(save_fname)
        time_ticks.append(time.time())
        time_messages.append('save_parameters')

        results = self.infer(data_iterator=val_iterator, log_interval=log_interval)
        time_ticks.append(time.time())
        time_messages.append('inference')

        # all gather results from all ranks
        if hvd.size() > 1:
            results = self.allgather(results)
        time_ticks.append(time.time())
        time_messages.append('allgather')

        # convert to numpy (cocoapi doesn't take mxnet ndarray)
        results = results.asnumpy()
        time_ticks.append(time.time())
        time_messages.append('asnumpy')

        time_ticks = np.array(time_ticks)
        elpased_time = time_ticks[1:]-time_ticks[:-1]
        validation_log_msg = '[Validation] '
        for msg, t in zip(time_messages, elpased_time):
            validation_log_msg += f'{msg}: {t*1000.0:.3f} [ms], '
        # TODO(ahmadki): val size is hard coded :(
        validation_log_msg += f'speed: {5000.0/(time_ticks[-1]-time_ticks[0]):.3f} [imgs/sec]'

        # TODO(ahmadki): remove time measurements
        logging.info(validation_log_msg)

        # Evaluate(score) results
        map_score = -1
        if self.async_executor:
            if hvd.rank() == 0:
                self.async_executor.submit(tag=str(epoch),
                                           fn=coco_map_score,
                                           results=results,
                                           annotation_file=annotation_file,
                                           num_threads=cocoapi_threads)
                def log_callback(future):
                    log_end(key=mlperf_constants.EVAL_STOP,
                            metadata={'epoch_num': epoch})
                    log_event(key=mlperf_constants.EVAL_ACCURACY,
                              value=future.result()/100,
                              metadata={'epoch_num': epoch})
                self.async_executor.add_done_callback(tag=str(epoch), fn=log_callback)
        else:
            if hvd.rank() == 0:
                map_score = coco_map_score(results=results,
                                           annotation_file=annotation_file,
                                           num_threads=cocoapi_threads)
            map_score = comm.bcast(map_score, root=0)
            log_end(key=mlperf_constants.EVAL_STOP,
                    metadata={'epoch_num': epoch})
            log_event(key=mlperf_constants.EVAL_ACCURACY,
                      value=map_score/100,
                      metadata={'epoch_num': epoch})
        return map_score

    def infer(self, data_iterator, log_interval=None):
        results = []
        images_counter = 0
        tick = time.time()
        for i, (images, shapes, ids) in enumerate(data_iterator):
            results.append(self.infer_net(images, self.anchors_mx, shapes, ids))
            images_counter += images.shape[0]
            if log_interval and not (i+1) % log_interval:
                mx.nd.waitall()
                tock = time.time()
                logging.info((f'[Inference][Iteration {i+1}] '
                              f'time: {(tock-tick)*1000.0:.3f} [ms], '
                              f'speed: {images_counter/(tock-tick):.3f} [imgs/gpu/sec]'))
                images_counter = 0
                tick = time.time()

        # Concatenate all batches
        results = mx.nd.concat(*results, dim=0)
        return results

    def get_async_results(self, waitall=False):
        val_map = -1
        val_epoch = -1
        if hvd.rank() == 0:
            if waitall:
                results = self.async_executor.result()
            else:
                results = self.async_executor.pop_done()
            if results and len(results) > 0:
                # get highest mAP (in case multiple results are returned)
                val_epoch = max(results, key=results.get)
                val_map = results[val_epoch]

        val_map = comm.bcast(val_map, root=0)
        return val_epoch, val_map

    def load_parameters(self, filename):
        logging.info(f'Loading pretrained network from {filename}')
        self.net.load_parameters(filename)

    def load_pretrain_backbone(self, picklefile_name):
        logging.info(f'Loading backbones weights from {picklefile_name}')
        pretrain_backbone(param_dict=self.net.features.collect_params(),
                          picklefile_name=picklefile_name,
                          layout=self.net.layout)

    def allgather(self, array):
        if self.ALL_GATHER_WITH_MPI4PY:
            array = array.copyto(mx.cpu())
            array = comm.allgather(array) # Do All gather
            array = mx.nd.concat(*array, dim=0) # Concatenate all ranks
        else:
            array = hvd.allgather(array)
        return array

    def allreduce_running(self):
        # allreduce running BN means and vars
        if hvd.size() > 1:
            for param_name, param in self.net.collect_params().items():
                if any(running_param in param_name for running_param in self.RUNNING_PARAMS):
                    t = param.data(ctx=self.ctx)
                    t = hvd.allreduce(t, average=True, name=None, priority=0)
                    param.set_data(t)

    def broadcast_params(self, root_rank=0):
        if hvd.size() > 1:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=root_rank)

    def zero_grads(self):
        for param_name, param in self.train_net.collect_params().items():
            param.zero_grad()

    def print_array_stats(self, name, array):
        print(f"{name:55}: {str(array.shape):20}, {np.min(array):>12f}, {np.max(array):>12f}, "
              f"{np.mean(array):>12f}, {np.std(array):>12f}")

    def print_params_stats(self, net):
        # Print resnet weights first (loaded from a file), then SSD weights
        print("\n\n")
        for param_name, param in net.collect_params().items():
            if 'resnetmlperf' in param_name:
                data = param.data().asnumpy()
                self.print_array_stats(param_name, data)
        print("\n\n")
        for param_name, param in net.collect_params().items():
            if 'resnetmlperf' not in param_name:
                if 'convpredictor' in param_name:
                    # print conf and loc separately
                    arr = param.data().asnumpy()
                    pos = param_name.find("convpredictor") + 13
                    layer_id = int(param_name[pos]) # layer number
                    offsets = net.predictor_offsets[layer_id]
                    categories = ["conf", "loc"]
                    for i in range(len(offsets)-1):
                        if i == 2: break # skip padding
                        begin, end = offsets[i], offsets[i+1]
                        data = arr[begin:end]
                        param_name_ = param_name.replace('convpredictor', categories[i])
                        self.print_array_stats(param_name_, data)
                else:
                    data = param.data().asnumpy()
                    self.print_array_stats(param_name, data)
        print("\n\n")

    @property
    def bn_group(self):
        return self.net.bn_group
