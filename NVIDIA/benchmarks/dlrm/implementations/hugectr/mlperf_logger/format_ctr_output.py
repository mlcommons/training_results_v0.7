import sys
import json
import argparse

from . import utils as mllogger
import mlperf_logging.mllog as mllog


def log_config(config):
    mllogger.mlperf_submission_log('dlrm')

    mllogger.log_event(key='eval_samples',
                       value=config['layers'][0]['eval_num_samples'])
    mllogger.log_event(key='global_batch_size',
                       value=config['solver']['batchsize'])
    mllogger.log_event(key='opt_base_learning_rate',
                       value=config['optimizer']['sgd_hparam']['learning_rate'])
    mllogger.log_event(key='sgd_opt_base_learning_rate',
                       value=config['optimizer']['sgd_hparam']['learning_rate'])
    mllogger.log_event(key='sgd_opt_learning_rate_decay_poly_power',
                       value=config['optimizer']['sgd_hparam'].get('decay_power'))
    mllogger.log_event(key='opt_learning_rate_warmup_steps',
                       value=config['optimizer']['sgd_hparam']['warmup_steps'])
    mllogger.log_event(key='opt_learning_rate_warmup_factor',
                       value=0.0) # not configurable
    mllogger.log_event(key='lr_decay_start_steps',
                       value=config['optimizer']['sgd_hparam'].get('decay_start'))
    mllogger.log_event(key='sgd_opt_learning_rate_decay_steps',
                       value=config['optimizer']['sgd_hparam'].get('decay_steps'))

class LogConverter:
    def __init__(self, steps_per_epoch, start_timestamp):
        self.start_time = start_timestamp
        self.steps_per_epoch = steps_per_epoch


    def _get_log_foo(self, key):
        if '_start' in key:
            return mllogger.log_start
        if '_end' in key or '_stop' in key:
            return mllogger.log_end
        else:
            return mllogger.log_event


    def _get_value(self, data):
        if data[0] == 'eval_accuracy':
            return float(data[1])
        if data[0] == 'train_samples':
            return int(data[1])


    def _get_metadata(self, data):
        if data[0] == 'eval_accuracy':
            self._last_eval_accuracy = float(data[1])
            return { 'epoch_num': float(data[2]) + 1 }
        if 'eval' in data[0]:
            return { 'epoch_num': float(data[1]) + 1 }
        if 'epoch' in data[0]:
            return { 'epoch_num': int(data[1]) + 1 }
        if data[0] == 'run_stop':
            return { 'status': 'success' if self._last_eval_accuracy > 0.8025 else 'aborted' }


    def _get_kvm(self, data):
        key = data[0]
        if data[0] == 'init_end':
            key = 'init_stop'
        if data[0] == 'train_epoch_start':
            key = 'epoch_start'
        if data[0] == 'train_epoch_end':
            key = 'epoch_stop'

        value = self._get_value(data)
        metadata = self._get_metadata(data)
        
        return key, value, metadata


    def _get_time_ms(self, ms):
        return self.start_time + int(float(ms))


    def validate_event(self, event):
        try:
            float(event[0])

            if not event[1].isidentifier():
                return False

            for x in event[2:]:
                float(x)
            return True
        except:
            return False


    def log_event(self, event_log):

        if self.validate_event(event_log):
            log_foo = self._get_log_foo(event_log[1])
            key, value, metadata = self._get_kvm(event_log[1:])
            time_ms = self._get_time_ms(event_log[0])

            log_foo(key=key, value=value, metadata=metadata, time_ms=time_ms)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('start_timestamp', type=int)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)
    log_config(config)

    converter = LogConverter(
        steps_per_epoch=(config['layers'][0]['num_samples']/config['solver']['batchsize']),
        start_timestamp=args.start_timestamp,
    )

    for line in sys.stdin:
        event_log = [x.strip() for x in line.strip().strip('][\x08 ,').split(',')]
        converter.log_event(event_log)


if __name__ == '__main__':
    main()
