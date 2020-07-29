from bisect import bisect

from mlperf_log_utils import log_event
from mlperf_logging.mllog import constants as mlperf_constants


class MLPerfLearningRateScheduler:
    _MLPERF_BASE_LR = 2.5e-3
    def __init__(self, learning_rate=_MLPERF_BASE_LR,
                 decay_factor=None, decay_epochs=None,
                 warmup_factor=None, warmup_epochs=None,
                 epoch_size=None, global_batch_size=None):
        if decay_epochs:
            assert (decay_factor is not None), 'decay_factor can\'t be None when decay_epochs is given'
        if warmup_epochs:
            assert (warmup_factor is not None), 'warmup_factor can\'t be None when warmup_epochs is given'

        self.lr = learning_rate or self._MLPERF_BASE_LR
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.epoch_size = epoch_size
        self.global_batch_size = global_batch_size

        # convert warmup epochs to iterations
        self.warmup_iters = None
        if self.warmup_epochs:
            self.warmup_iters = int(self.warmup_epochs * self.epoch_size / self.global_batch_size)
            log_event(key=mlperf_constants.OPT_LR_WARMUP_STEPS, value=self.warmup_iters)
            log_event(key=mlperf_constants.OPT_LR_WARMUP_FACTOR, value=self.warmup_factor)

        self.lr = self.mlperf_adjusted_lr(requested_lr=self.lr,
                                          global_batch_size=global_batch_size)

        log_event(key=mlperf_constants.OPT_BASE_LR, value=self.lr)
        log_event(key=mlperf_constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=self.decay_epochs)

    # the base_lr from the mlperf reference must be scaled by an integer.  But we
    # also want to scale by approximately global_batch_size/32, all else being
    # equal, so we choose an appropriate integer.
    # https://github.com/mlperf/training_policies/blob/master/training_rules.adoc#hyperparameters-and-optimizer
    def mlperf_adjusted_lr(self, requested_lr, global_batch_size):
        requested_lr_multiplier = requested_lr / self._MLPERF_BASE_LR
        actual_lr_multiplier = max(1, round(requested_lr_multiplier * global_batch_size / 32))
        return self._MLPERF_BASE_LR * actual_lr_multiplier


    def __call__(self, current_epoch=None, current_iter=None):
        current_lr = self.lr

        # learning rate decay
        if self.decay_factor and self.decay_epochs:
            current_lr *= (self.decay_factor ** bisect(self.decay_epochs, current_epoch-1))

        # learning warmup
        if self.warmup_iters and self.warmup_iters > current_iter:
            warmup_step = current_lr / (self.warmup_iters * (2 ** self.warmup_factor))
            current_lr -= (self.warmup_iters - current_iter) * warmup_step

        return current_lr
