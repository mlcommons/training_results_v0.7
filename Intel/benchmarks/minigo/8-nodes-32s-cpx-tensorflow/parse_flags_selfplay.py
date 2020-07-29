from absl import app, flags
from mlperf_logging import mllog

import os

flags.DEFINE_integer('num_readouts', None,
                    'num_readouts')
flags.DEFINE_float('fastplay_frequency', None,
                    'fastpaly frequency')
flags.DEFINE_integer('fastplay_readouts', None,
                    'fastplay_readouts')
flags.DEFINE_float('value_init_penalty', None, 
                    'value init penalty')
flags.DEFINE_float('holdout_pct', None, 
                    'holdout pct')
flags.DEFINE_float('disable_resign_pct', None, 
                    'disable resign pct')
flags.DEFINE_float('min_resign_threshold', None, 
                    'min resign threshold')
flags.DEFINE_float('max_resign_threshold', None, 
                    'max resign threshold')
flags.DEFINE_integer('virtual_losses', None, 
                    'virtual losses')
flags.DEFINE_float('dirichlet_alpha', None, 
                    'dirichlet_alpha')
flags.DEFINE_float('noise_mix', None, 
                    'noise mix')
flags.DEFINE_integer('cache_size_mb', None, 
                    'cache size')
flags.DEFINE_bool('verbose', None,
                    'verbose')
flags.DEFINE_integer('selfplay_threads', None,
                    'selfplay threads')
flags.DEFINE_integer('parallel_search', None,
                    'parallel search')
flags.DEFINE_integer('parallel_inference', None,
                    'parallel inference')
flags.DEFINE_integer('concurrent_games_per_thread', None,
                    'concurrent games per thread')
flags.DEFINE_integer('target_pruning', None,
                    'target pruning')



FLAGS = flags.FLAGS

def main(argv):
    mllogger = mllog.get_mllogger()
    mllog.config(filename="train.log")

    mllog.config(
      default_namespace = "worker1",
      default_stack_offset = 1,
      default_clear_line = False,
      root_dir = os.path.normpath("/tmp/"))


    mllogger.event(key='num_readouts', value=FLAGS.num_readouts)
    mllogger.event(key='value_init_penalty', value=FLAGS.value_init_penalty)
    mllogger.event(key='holdout_pct', value=FLAGS.holdout_pct)
    mllogger.event(key='disable_resign_pct', value=FLAGS.disable_resign_pct)
    mllogger.event(key='min_resign_threshold', value=FLAGS.min_resign_threshold)
    mllogger.event(key='max_resign_threshold', value=FLAGS.max_resign_threshold)
    mllogger.event(key='selfplay_threads', value=FLAGS.selfplay_threads)
    mllogger.event(key='parallel_games', value=FLAGS.parallel_inference)
    mllogger.event(key='virtual_losses', value=FLAGS.virtual_losses)
    
if __name__ == '__main__':
    app.run(main)
