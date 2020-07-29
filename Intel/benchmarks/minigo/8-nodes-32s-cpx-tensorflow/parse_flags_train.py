from absl import app, flags
from mlperf_logging import mllog
import os

flags.DEFINE_multi_float('lr_rates', None,
                    'lr rates')
flags.DEFINE_multi_float('lr_boundaries', None,
                    'learning rate boundaries')
flags.DEFINE_float('l2_strength', None,
                    'weight decay')
flags.DEFINE_integer('conv_width', None, 
                    'conv width')
flags.DEFINE_integer('fc_width', None, 
                    'fc width')
flags.DEFINE_integer('trunk_layers', None, 
                    'trunk layers')
flags.DEFINE_float('value_cost_weight', None, 
                    'value cost weight')
flags.DEFINE_integer('summary_steps', None, 
                    'summary steps')
flags.DEFINE_integer('bool_features', None, 
                    'bool features')
flags.DEFINE_string('input_features', None, 
                    'input features')
flags.DEFINE_string('input_layout', None, 
                    'input layout')
flags.DEFINE_integer('shuffle_buffer_size', None,
                    'shuffle buffer size')
flags.DEFINE_boolean('shuffle_examples', None,
                    'shuffle examples')
flags.DEFINE_integer('keep_checkpoint_max', None,
                    'keep_checkpoint_max')
flags.DEFINE_integer('train_batch_size', None,
                    'train_batch_size')

FLAGS = flags.FLAGS

def main(argv):
    mllogger = mllog.get_mllogger()
    mllog.config(filename="train.log")

    mllog.config(
      default_namespace = "worker1",
      default_stack_offset = 1,
      default_clear_line = False)


    mllogger.event(key=mllog.constants.OPT_BASE_LR, value=FLAGS.lr_rates)
    mllogger.event(key='lr_rates', value=FLAGS.lr_rates)
    mllogger.event(key=mllog.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=FLAGS.lr_boundaries[1])
    mllogger.event(key='lr_boundaries', value=FLAGS.lr_boundaries[1])
    mllogger.event(key=mllog.constants.OPT_WEIGHT_DECAY, value=FLAGS.l2_strength)
    mllogger.event(key='opt_learning_rate_decay_boundary_steps', value=FLAGS.lr_boundaries)
    mllogger.event(key='train_batch_size', value=FLAGS.train_batch_size)
    
if __name__ == '__main__':
    app.run(main)
