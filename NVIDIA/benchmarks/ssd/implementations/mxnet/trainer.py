import horovod.mxnet as hvd
import mxnet as mx
from mxnet.contrib import amp

from mlperf_log_utils import log_event
from mlperf_logging.mllog import constants as mlperf_constants

def sgd_trainer(net, learning_rate, weight_decay, momentum, precision,
                fp16_loss_scale, gradient_predivide_factor, num_groups,
                profile_no_horovod):
    # Trainer
    if not profile_no_horovod:
        trainer = hvd.DistributedTrainer(net.collect_params(), 'sgd',
                                         {'learning_rate': learning_rate,
                                          'wd': weight_decay,
                                          'momentum': momentum,
                                          'multi_precision': precision == 'fp16',
                                          'rescale_grad':1.0/fp16_loss_scale if precision == 'fp16' else 1.0},
                                         gradient_predivide_factor=gradient_predivide_factor,
                                         num_groups=num_groups)
    else:
        trainer = mx.gluon.Trainer(net.collect_params(), 'sgd',
                                   {'learning_rate': learning_rate,
                                    'wd': weight_decay,
                                    'momentum': momentum,
                                    'multi_precision': precision == 'fp16',
                                    'rescale_grad':1.0/fp16_loss_scale if precision == 'fp16' else 1.0},
                                   kvstore=None)
    log_event(key=mlperf_constants.OPT_WEIGHT_DECAY, value=weight_decay)
    if precision == 'amp':
        amp.init_trainer(trainer)

    return trainer
