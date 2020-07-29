from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor
import time
import datetime
import numpy as np
from mlperf_logging import mllog

class StateMonitor(Callback):
    def __init__(self, data_size, mllogger, tot_batch_size=None, lrs=None, device_id=0, model=None, eval_dataset=None, eval_interval=None):
        super(StateMonitor, self).__init__()
        self.data_size = data_size
        self.tot_batch_size = tot_batch_size
        self.lrs = lrs
        self.device_id = device_id
        self.epoch_num = 0
        self.loss = 0
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_interval = eval_interval
        self.mllogger = mllogger

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.loss = loss

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()
        self.mllogger.start(key=mllog.constants.BLOCK_START, metadata={"first_epoch_num": self.epoch_num + 1, "epoch_count": self.eval_interval})

    def epoch_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time)
        per_step_seconds = epoch_seconds / self.data_size

        print_str = "epoch[{}]".format(self.epoch_num)
        print_str += ', epoch time: {:.2f}s'.format(epoch_seconds)
        print_str += ', per step time: {:.4f}s'.format(per_step_seconds)
        print_str += ', loss={:.6f}'.format(self.loss)

        if self.lrs is not None:
            lr = self.lrs[int(self.epoch_num // self.eval_interval + 1) * self.data_size - 1]
            print_str += ', lr={:.6f}'.format(lr)

        if self.tot_batch_size is not None:
            fps = self.tot_batch_size * self.data_size / epoch_seconds
            print_str += ', fps={:.2f}'.format(fps)

        eval_start = time.time()
        output = self.model.eval(self.eval_dataset)
        eval_seconds = time.time() - eval_start
        self.mllogger.end(key=mllog.constants.BLOCK_STOP, metadata={"first_epoch_num": self.epoch_num + 1})
        self.mllogger.event(key=mllog.constants.EVAL_ACCURACY, value=float(output["acc"]), metadata={"epoch_num": self.epoch_num + 4})

        print_str += ', accuracy={:.6f}'.format(float(output["acc"]))
        print_str += ', eval_cost={:.2f}'.format(eval_seconds)
        if self.device_id == 0:
            print(print_str, flush=True)
        
        if float(output["acc"]) > 0.759:
            run_context.request_stop()

        self.epoch_num += self.eval_interval
