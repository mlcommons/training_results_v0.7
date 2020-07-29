# Copyright 2019 MLBenchmark Group. All Rights Reserved.
#
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
# ==============================================================================

"""Utilities for compliance logging."""

import logging
import time
import inspect
import sys
import subprocess
from mlperf_logging.mllog import constants as mlperf_constants

class MPIWrapper(object):
    def __init__(self):
        self.mx = None
        self.hvd = None
        try:
            import mxnet as mx
            self.mx = mx
        except ImportError:
            raise ImportError("Failed to import MXNet, is it installed properly?")

        try:
            import horovod.mxnet as hvd
            self.hvd = hvd
            self.hvd.rank()
        except ImportError:
            raise ImportError("Failed to import Horovod, is it installed properly?")
        except ValueError:
            self.hvd.init()
    
    def barrier(self):
        allreduce(1.)

    def allreduce(self, x):
        if self.hvd:
            if not isinstance(x, self.mx.ndarray.NDArray):
                array = self.mx.nd.array([x])
            else:
                array = x
            self.hvd.allreduce_(array, average=False, name="allreduce")
            return array.asnumpy()
        return x

    def get_rank(self):
        if self.hvd:
            return self.hvd.rank()
        return 0


mpiwrapper = MPIWrapper()

def allreduce(x):
    return mpiwrapper.allreduce(x)

def print_submission_info():
    def query(command):
        result = subprocess.check_output(
            command, shell=True,
            stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        return result

    log(mlperf_constants.SUBMISSION_ORG, val='Intel')
    # hardware = query('cat /sys/devices/virtual/dmi/id/product_name')
    log(mlperf_constants.SUBMISSION_PLATFORM, val='{}'.format("one node with Cooperlake"))
    log(mlperf_constants.SUBMISSION_DIVISION, val='closed')
    log(mlperf_constants.SUBMISSION_BENCHMARK, val='resnet')
    log(mlperf_constants.SUBMISSION_STATUS, val='onprem')

def cache_clear():
    log(mlperf_constants.CACHE_CLEAR, val='true')

def init_start():
    log(mlperf_constants.INIT_START, caller_depth=3, event_type='INTERVAL_START')

def init_stop(sync=False):
    if sync:
        mpiwrapper.barrier()
    log(mlperf_constants.INIT_STOP, caller_depth=3, event_type='INTERVAL_END')

def run_start(sync=False):
    if sync:
        mpiwrapper.barrier()
    log(mlperf_constants.RUN_START, caller_depth=3, event_type='INTERVAL_START')

def run_stop(status):
    assert status == 'success' or status == 'aborted'
    log(mlperf_constants.RUN_STOP,
      meta_data = {'status': status},
      caller_depth=3, event_type='INTERVAL_END')

def block_start(epoch, count):
    log(mlperf_constants.BLOCK_START,
      meta_data = {'first_epoch_num':epoch,
                   'epoch_count': count},
      caller_depth=3, event_type='INTERVAL_START')

def block_stop(epoch):
    log(mlperf_constants.BLOCK_STOP,
      meta_data = {'first_epoch_num': epoch},
      caller_depth=3, event_type='INTERVAL_END')

def epoch_start(epoch):
    log(mlperf_constants.EPOCH_START,
      meta_data = {'epoch_num': epoch},
      caller_depth=3, event_type='INTERVAL_START')

def epoch_stop(epoch):
    log(mlperf_constants.EPOCH_STOP,
      meta_data = {'epoch_num': epoch},
      caller_depth=3, event_type='INTERVAL_END')

def eval_start(epoch, sync=False):
    if sync:
        mpiwrapper.barrier()
    log(mlperf_constants.EVAL_START,
      meta_data = {'epoch_num': epoch},
      caller_depth=3, event_type='INTERVAL_START')

def eval_stop(epoch):
    log(mlperf_constants.EVAL_STOP,
      meta_data = {'epoch_num': epoch},
      caller_depth=3, event_type='INTERVAL_END')

def eval_accuracy(epoch, accuracy):
    log(mlperf_constants.EVAL_ACCURACY,
      val = '{}'.format(accuracy),
      meta_data = {'epoch_num': epoch},
      caller_depth=3)

def seed(seed):
    log('seed',
      val = '{}'.format(seed),
      caller_depth=3)

def training_samples(num_training_samples):
    log('training_samples',
      val = '{}'.format(num_training_samples),
      caller_depth=3)

def evaluation_samples(num_eval_samples):
    log('evaluation_samples',
      val = '{}'.format(num_eval_samples),
      caller_depth=3)

def epoch_training_samples(epoch, epoch_training_samples):
    log('epoch_training_samples',
      val = '{}'.format(epoch_training_samples),
      meta_data = {'epoch_num': epoch},
      caller_depth=3)

def global_batch_size(batch_size):
    log(mlperf_constants.GLOBAL_BATCH_SIZE,
      val = '{}'.format(batch_size),
      caller_depth=3)

def model_bn_span(span):
    log(mlperf_constants.MODEL_BN_SPAN,
      val = '{}'.format(span),
      caller_depth=3)

def opt_name(opt):
    if opt == "sgdwfastlars":
        opt = "lars"
    else:
        opt = "sgd"
    log(mlperf_constants.OPT_NAME,
      val = '{}'.format(opt),
      caller_depth=3)

def opt_weight_decay(weight_decay):
    log(mlperf_constants.OPT_WEIGHT_DECAY,
      val = '{}'.format(weight_decay),
      caller_depth=3)

def lr_rates(rates):
    log(mlperf_constants.OPT_BASE_LR,
      val = '{}'.format(rates),
      caller_depth=3)

def warmup_epoch(warmup_epochs):
    log(mlperf_constants.OPT_LR_WARMUP_EPOCHS,
      val = '{}'.format(warmup_epochs),
      caller_depth=3)

def lr_boundary_epochs(boundary_epoch):
    log(mlperf_constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
      val = '{}'.format(boundary_epoch),
      caller_depth=3)

def lars_epsilon(epsilon):
    log(mlperf_constants.LARS_EPSILON,
      val = '{}'.format(epsilon),
      caller_depth=3)

def lars_opt_end_lr(lars_end_lr):
    log(mlperf_constants.LARS_OPT_END_LR,
      val = '{}'.format(lars_end_lr),
      caller_depth=3)

def lars_opt_lr_decay_steps(decay_steps):
    log(mlperf_constants.LARS_OPT_LR_DECAY_STEPS,
      val = '{}'.format(decay_steps),
      caller_depth=3)

def lars_opt_lr_decay_poly_power(power):
    log('lars_opt_learning_rate_decay_poly_power',
      val = '{}'.format(power),
      caller_depth=3)

def lars_opt_base_learning_rate(lr):
    log("lars_opt_base_learning_rate",
      val = '{}'.format(lr),
      caller_depth=3)

def lars_opt_weight_decay(wd):
    log(mlperf_constants.LARS_OPT_WEIGHT_DECAY,
      val = '{}'.format(wd),
      caller_depth=3)

def lars_opt_learning_rate_warmup_epochs(warmup_epoch):
    log("lars_opt_learning_rate_warmup_epochs",
      val = '{}'.format(warmup_epoch),
      caller_depth=3)

def lars_opt_momentum(momentum):
    log("lars_opt_momentum",
      val = '{}'.format(momentum),
      caller_depth=3)

def lr_boundary_steps(boundary_step):
    log(mlperf_constants.OPT_LR_DECAY_BOUNDARY_STEPS,
      val = '{}'.format(boundary_step),
      caller_depth=3)

def eval_result(iteration, timestamp):
    log('eval_result',
      meta_data = {'iteration': iteration, 'timestamp': timestamp},
      caller_depth=3)

def log(key, val='null', meta_data=None, caller_depth=2, event_type='POINT_IN_TIME'):
    if mpiwrapper.get_rank() == 0:
        filename, lineno = get_caller(caller_depth)
        meta_dict = {"lineno": lineno, "file": "%s" % filename}
        if meta_data != None:
            meta_dict.update(meta_data)
        meta_string = "{}".format(meta_dict)
        if val == 'null' or val[0].isdigit():
            print(':::MLLOG {"namespace": %d, "time_ms": %f, "event_type": "%s", "key": "%s", "value": %s, "metadata": %s}' % 
                    (0, time.time()*1000.0, event_type, key, val, meta_string), file=sys.stderr)
        else:
            print(':::MLLOG {"namespace": %d, "time_ms": %f, "event_type": "%s", "key": "%s", "value": "%s", "metadata": %s}' %
                    (0, time.time()*1000.0, event_type, key, val, meta_string), file=sys.stderr)

def get_caller(stack_index=2, root_dir=None):
    ''' Returns file.py:lineno of your caller. A stack_index of 2 will provide
        the caller of the function calling this function. Notice that stack_index
        of 2 or more will fail if called from global scope. '''
    caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

    # Trim the filenames for readability.
    filename = caller.filename
    if root_dir is not None:
        filename = re.sub("^" + root_dir + "/", "", filename)
    return (filename, caller.lineno)
