# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys
sys.path.insert(0, '.')  # nopep8

import itertools
import logging
import os
import tensorflow as tf
import time
from ml_perf.utils import *
from ml_perf.logger import log_event, log_start, log_end
from mlperf_logging.mllog import constants

from absl import app, flags
from mpi4py import MPI
import socket

import sys
import minigo_python
import math
import numpy as np

import train as minigo_train


flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_float('train_filter', 0.3,
                   'Fraction of selfplay games to pass to training.')

flags.DEFINE_integer('examples_per_generation', 131072,
                     'Number of examples use from each generation in the '
                     'training window.')

flags.DEFINE_boolean('validate', False, 'Run validation on holdout games')

flags.DEFINE_integer('min_games_per_iteration', 4096,
                     'Minimum number of games to play for each training '
                     'iteration.')

flags.DEFINE_integer('suggested_games_per_iteration', 8192,
                     'Suggested number of games to play for each training '
                     'iteration. Should be >= min_games_per_iteration')

flags.DEFINE_integer('num_read_threads', 64,
                     'Number of threads to read examples on. Using more '
                     'read threads may speed up reading the examples as '
                     'more can be decompressed in parallel. This flag has '
                     'no effect on the output data.')

flags.DEFINE_integer('num_write_threads', 8,
                     'Number of threads to write examples on. Each thread '
                     'will write a separate .tfrecord.zz file to train on. '
                     'Using more threads may reduce the time take to generate '
                     'the training chunks as more threads are used to '
                     'compress the data. Using too many threads however could '
                     'slow down training time if each shard gets much smaller '
                     'than around 100MB.')

# dir flags
flags.DEFINE_string('base_dir', None, 'Training base dir.')
flags.DEFINE_string('golden_chunk_dir', None, 'Training example directory.')
flags.DEFINE_string('holdout_dir', None, 'Holdout example directory.')
flags.DEFINE_string('model_dir', None, 'Model directory.')
flags.DEFINE_string('selfplay_dir', None, 'Selfplay example directory.')
flags.DEFINE_string('selfplay_log_dir', None, 'Selfplay log directory.')

# checkpoint path
flags.DEFINE_string('checkpoint_dir', None, 'Source checkpoint directory.')
flags.DEFINE_string('target_path', None, 'Full path to the target file for eval.')
flags.DEFINE_string('board_size', '19', 'Board size.')

# sample_records flags
flags.DEFINE_integer('num_records', 0, '.')
flags.DEFINE_boolean('compression', True, 'Enable/Disable compression.')
flags.DEFINE_boolean('shuffle', True, 'Enable/Disable shuffle.')


# selfplay flags
flags.DEFINE_integer('cache_size_mb', 8192, 'Cache size in MB.')
flags.DEFINE_integer('cache_shards', 8, 'Number of cache shards.')
flags.DEFINE_integer('num_readouts', 800, 'Readouts/move.')
flags.DEFINE_float('fastplay_frequency', 0.75, 'Fast play frequency.')
flags.DEFINE_integer('fastplay_readouts', 80, 'Fast play readouts.')
flags.DEFINE_integer('virtual_losses', 4, 'Fast play readouts.')
flags.DEFINE_float('dirichlet_alpha', 0.03, 'dirichlet alpha.')
flags.DEFINE_float('noise_mix', 0.3, 'noise_mix.')
flags.DEFINE_float('value_init_penalty', 0.2, 'value_init_penalty.')
flags.DEFINE_boolean('target_pruning', True, '')
flags.DEFINE_float('policy_softmax_temp', 0.98, 'value_init_penalty.')
flags.DEFINE_boolean('allow_pass', True, '')
flags.DEFINE_integer('restrict_pass_alive_play_threshold', 4, '')
flags.DEFINE_integer('selfplay_threads', 3, '')
flags.DEFINE_integer('parallel_search', 4, '')
flags.DEFINE_integer('parallel_inference', 2, '')
flags.DEFINE_integer('concurrent_games_per_thread', 32, '')
flags.DEFINE_float('min_resign_threshold', -1.0, '')
flags.DEFINE_float('max_resign_threshold', 0.9, '')
flags.DEFINE_float('disable_resign_pct', 0.0, '')
flags.DEFINE_integer('num_games', 0, 'Num games to run in selfplay, only works if run_forever is False')
flags.DEFINE_boolean('run_forever', True, 'Run selfplay in while(true) loop')
flags.DEFINE_float('holdout_pct', 0.03, 'Amount of holdout data in selfplay to use for validation.')
flags.DEFINE_boolean('selfplay_verbose', False, '')
flags.DEFINE_integer('output_threads', 1, '')

# eval
flags.DEFINE_integer('num_eval_games', 256, 'Number of games for eval.')
flags.DEFINE_integer('num_eval_readouts', 100, 'Eval Readouts/move.')
flags.DEFINE_boolean('resign_enabled', False, '')
flags.DEFINE_float('resign_threshold', -0.999, '')
flags.DEFINE_float('winrate', 0.5, 'Fraction of games that a model must beat the target by.')

# seed 
flags.DEFINE_integer('seed', 0, 'Random seed. Use default value of 0 to use a time-based seed.')

# minigo verbose
flags.DEFINE_boolean('verbose', True, '')

# rank allocation flags
flags.DEFINE_integer('num_gpus_train', 1, 'Train uses gpus 0 through num_gpus_train-1 on respective MPI ranks.')
flags.DEFINE_integer('procs_per_gpu', 1, 'MPI processes per gpu.')
flags.DEFINE_integer('rank_gpu_index', 0, 'GPU that this rank uses.')
flags.DEFINE_integer('ranks_per_node', 2, 'MPI ranks per node.')
flags.DEFINE_integer('num_nodes', 1, 'Number of nodes.')
flags.DEFINE_integer('num_train_nodes', 1, 'Number of nodes for training.')
flags.DEFINE_integer('num_selfplay_nodes', 1, 'Number of nodes for selfplay.')

# From train.py
flags.declare_key_flag('freeze')
flags.declare_key_flag('use_trt')
flags.declare_key_flag('trt_max_batch_size')
flags.declare_key_flag('trt_precision')
flags.declare_key_flag('shuffle_buffer_size')
flags.declare_key_flag('shuffle_examples')
flags.declare_key_flag('window_size')

# From dual_net.py
flags.declare_key_flag('work_dir')
flags.declare_key_flag('train_batch_size')
flags.declare_key_flag('lr_rates')
flags.declare_key_flag('lr_boundaries')
flags.declare_key_flag('l2_strength')
flags.declare_key_flag('conv_width')
flags.declare_key_flag('fc_width')
flags.declare_key_flag('trunk_layers')
flags.declare_key_flag('value_cost_weight')
flags.declare_key_flag('summary_steps')
flags.declare_key_flag('bool_features')
flags.declare_key_flag('input_features')
flags.declare_key_flag('input_layout')

FLAGS = flags.FLAGS


# Training loop state.
class State:
    def __init__(self, model_num):
        self.start_time = time.time()
        self.start_iter_num = model_num
        self.iter_num = model_num

    def _model_name(self, it):
        return '%06d' % it

    @property
    def selfplay_model_name(self):
        return self._model_name(self.iter_num - 1)

    @property
    def selfplay_model_path(self):
        return '{}.pb'.format(
            os.path.join(FLAGS.model_dir, self.selfplay_model_name))

    @property
    def train_model_name(self):
        return self._model_name(self.iter_num)

    @property
    def train_model_path(self):
        return '{}.pb'.format(
            os.path.join(FLAGS.model_dir, self.train_model_name))


def wait_for_training_examples(state, num_games, pattern_suffix='.zz'):
    """Wait for training examples to be generated by the latest model.

    Args:
        state: the RL loop State instance.
        num_games: number of games to wait for.
    """

    paths = []
    model_dir = os.path.join(FLAGS.selfplay_dir, state.selfplay_model_name)
    pattern = os.path.join(model_dir, '*', '*', '*.tfrecord' + pattern_suffix)
    logging.info('Will be looking for %d games in %s, iter %d', num_games, model_dir, state.iter_num)
    for i in itertools.count():
        try:
            paths = sorted(tf.gfile.Glob(pattern))
        except tf.errors.OpError:
            paths = []
        if len(paths) >= num_games:
            break
        if i % 30 == 0:
            logging.info('Waiting for %d games in %s (found %d)',
                         num_games, model_dir, len(paths))
        time.sleep(1)

    return paths


def list_selfplay_dirs(base_dir):
    """Returns a sorted list of selfplay data directories.

    Training examples are written out to the following directory hierarchy:
      base_dir/device_id/model_name/timestamp/

    Args:
      base_dir: either selfplay_dir or holdout_dir.

    Returns:
      A list of model directories sorted so the most recent directory is first.
    """

    model_dirs = [os.path.join(base_dir, x)
                  for x in tf.io.gfile.listdir(base_dir)]
    return sorted(model_dirs, reverse=True)


def sample_training_examples(state):
    """Sample training examples from recent selfplay games.

    Args:
        state: the RL loop State instance.

    Returns:
        A (num_examples, record_paths) tuple:
         - num_examples : number of examples sampled.
         - record_paths : list of golden chunks up to window_size in length,
                          sorted by path.
    """

    # Read examples from the most recent `window_size` models.
    model_dirs = list_selfplay_dirs(FLAGS.selfplay_dir)[:FLAGS.window_size]
    src_patterns = [os.path.join(x, '*', '*', '*.tfrecord.zz')
                    for x in model_dirs]

    # log games per generation used
    games_this_window = [len(tf.gfile.Glob(pattern)) for pattern in src_patterns]
    for game in games_this_window:
        log_event(key='actual_selfplay_games_per_generation', value=game, metadata={'epoch_num': state.iter_num})

    dst_path = os.path.join(FLAGS.golden_chunk_dir,
                            '{}.tfrecord.zz'.format(state.train_model_name))

    logging.info('Writing training chunks to %s', dst_path)

    # multi-gpu change write threads to be exactly number of gpus
    FLAGS.num_write_threads = 4 * FLAGS.num_gpus_train

    # selfplay records are sent over to train
    # sample_records reads everything, filters, shuffles, and writes
    num_examples = minigo_python.run_sample_records(FLAGS.train_filter,
                                                    FLAGS.num_records,
                                                    FLAGS.num_read_threads,
                                                    FLAGS.num_write_threads,
                                                    FLAGS.compression,
                                                    FLAGS.min_games_per_iteration,
                                                    FLAGS.shuffle,
                                                    FLAGS.seed,
                                                    src_patterns,
                                                    dst_path,
                                                    FLAGS.verbose)

    chunk_pattern = os.path.join(
        FLAGS.golden_chunk_dir,
        '{}-*-of-*.tfrecord.zz'.format(state.train_model_name))
    chunk_paths = sorted(tf.gfile.Glob(chunk_pattern))
    assert len(chunk_paths) == FLAGS.num_write_threads

    return (num_examples, chunk_paths)


def sample_checkpoint_examples(rank):
    """Preprocess checkpoint files to sample FLAGS.train_filter elements
    """

    if rank >= FLAGS.ranks_per_node:
        return

    # Read examples from the most recent `window_size` models.
    model_dirs = list_selfplay_dirs(os.path.join(FLAGS.checkpoint_dir, 'data/selfplay'))[:FLAGS.window_size]

    for x in model_dirs:
        src_pattern = os.path.join(x, '*', '*', '*.tfrecord.zz')
        paths = sorted(tf.gfile.Glob(src_pattern))
        mypaths = paths[rank::FLAGS.ranks_per_node]
        logging.info('rank %d is sampling %d paths at %s', rank, len(mypaths), x)

        for path in mypaths:
            dst_path = path.replace(os.path.join(FLAGS.checkpoint_dir, 'data/selfplay'), FLAGS.selfplay_dir)
            num_examples = minigo_python.run_sample_records(FLAGS.train_filter,
                                                            FLAGS.num_records,
                                                            1,
                                                            1,
                                                            FLAGS.compression,
                                                            0,
                                                            1,
                                                            FLAGS.seed,
                                                            [path],
                                                            dst_path,
                                                            False)

    logging.info('rank %d is done with sampling checkpoint paths', rank)
    return


def append_timestamp(elapsed, model_name):
  # Append the time elapsed from when the RL was started to when this model
  # was trained. GCS files are immutable, so we have to do the append manually.
  timestamps_path = os.path.join(FLAGS.model_dir, 'train_times.txt')
  try:
    with tf.gfile.Open(timestamps_path, 'r') as f:
      timestamps = f.read()
  except tf.errors.NotFoundError:
      timestamps = ''
  timestamps += '{:.3f} {}\n'.format(elapsed, model_name)
  with tf.gfile.Open(timestamps_path, 'w') as f:
      f.write(timestamps)


def sync_train_nodes(state, record_paths, tzcomm):
  # Sync before start
  tzcomm.barrier()
  trank  = tzcomm.Get_rank()
  tsize  = tzcomm.Get_size()

  record_size = len(record_paths)
  node_chunk_size  = record_size // tsize
  rank_chunk_size  = record_size // FLAGS.num_gpus_train
  rindex_start = FLAGS.rank_gpu_index * rank_chunk_size
  rindex_end   = rindex_start + rank_chunk_size

  logging.info('sync_train_nodes data transfer iter %s trank %d rindex %d', state.iter_num, trank, FLAGS.rank_gpu_index)

  if trank == 0:
     for i in range(rindex_start, rindex_end):
         send_bytes_list = []*rank_chunk_size
         for record in record_paths[i:record_size:node_chunk_size]:
            with open(record, 'rb') as f:
              send_bytes = f.read()
              send_bytes_list.append(send_bytes)
         logging.info('trank {}, dest={} records {} send_bytes_list {}'.format(trank, i, record_paths[i:record_size:node_chunk_size], list(map(len, send_bytes_list))))
         tzcomm.scatter(send_bytes_list, root=0)
         tzcomm.barrier()
  else:
     for i in range(rindex_start, rindex_end):
         recv_bytes = tzcomm.scatter(None, root=0)
         windex = i + trank * node_chunk_size
         with open(record_paths[windex], 'wb') as f:
             f.write(recv_bytes)
         logging.info('irecv trank {} record {} recv_bytes {}'.format(trank, record_paths[windex], len(recv_bytes)))
         tzcomm.barrier()


def get_training_input(state, rank, tcomm, tzcomm):
    """sample records, and distribute data on all train nodes.
    Args:
        state: the RL loop State instance.
    """

    num_examples = 0
    record_paths = []

    # wait for sample records
    if rank == 0:
        if state.iter_num > 22:
            games_needed = FLAGS.suggested_games_per_iteration
        else:
            games_needed = FLAGS.min_games_per_iteration
        wait_for_training_examples(state, games_needed)
        with logged_timer('[rank {}] sample records train: {}'.format(rank, state.iter_num)):
            num_examples, record_paths = sample_training_examples(state)

    tcomm.barrier()
    num_examples = tcomm.bcast(num_examples, root=0)
    record_paths = tcomm.bcast(record_paths, root=0)

    #sync train nodes for multinode training
    if FLAGS.num_train_nodes > 1:
        nodestr = 'node-' + str(rank // FLAGS.ranks_per_node)
        record_paths = [ s.replace('node-0', nodestr) for s in record_paths]
        with logged_timer('[rank {}] scatter train: {}'.format(rank, state.iter_num)):
            sync_train_nodes(state, record_paths, tzcomm)

    #
    tcomm.barrier()

    # each rank should see only the share of record paths that
    # are present on its node.
    chunk_size  = len(record_paths) // FLAGS.num_train_nodes
    chunk_start = (rank // FLAGS.ranks_per_node) * chunk_size
    chunk_end   = chunk_start + chunk_size

    return num_examples, record_paths[chunk_start:chunk_end]

def run_train(state, rank, tstate, tcomm, tzcomm):
    """Run training and write a new model to the model_dir.

    Args:
        state: the RL loop State instance.
    """

    # sample records from selfplay.
    # distribute sampled records across training nodes if multinode training.
    num_examples, record_paths = get_training_input(state, rank, tcomm, tzcomm)

    # train
    with logged_timer('[rank {}] Training: {}'.format(rank, state.iter_num)):
        minigo_train.run(state, rank, tstate, num_examples, record_paths)

    if rank == 0:
        # Append the time elapsed from when the RL was started to when this model
        # was trained.
        elapsed = time.time() - state.start_time
        append_timestamp(elapsed, state.train_model_name)
        log_event(key='save_model', value={'iteration': state.iter_num})


def transfer_file_to_selfplay(state, dcomm, filename):
  # Sync before start
  dcomm.barrier()
  drank  = dcomm.Get_rank()

  # selfplay nodes get file from training rank
  # training saves model on rank-0/node-0
  #  ==> data-transfer drank-0 (which is the 17th rank on node-0) sends
  #  ==> data-transfer dranks (the 17th rank) on all other nodes recv.
  if drank == 0:
    with open(filename, 'rb') as f:
        send_bytes = f.read()
    recv_bytes = dcomm.bcast(send_bytes, root=0)
    logging.info('Broadcast train d-rank %d iter %s send_bytes[%s] %d recv_bytes %d', drank, state.iter_num, filename, len(send_bytes), len(recv_bytes))
  else:
    recv_bytes = dcomm.bcast(None, root=0)
    with open(filename, 'wb') as f:
      f.write(recv_bytes)
      logging.info('Broadcast d-rank %d iter %s NO send_bytes (pb) num_recv_bytes %d', drank, state.iter_num, len(recv_bytes))


def transfer_trained_model_to_selfplay(state, dcomm):
  # local rank
  drank  = dcomm.Get_rank()
  model_path        = os.path.join(FLAGS.model_dir, state.train_model_name + '.minigo')
  eval_model_path   = os.path.join(FLAGS.model_dir, state.train_model_name + '.evalfp32minigo')
  model_path_staged = os.path.join(FLAGS.model_dir, state.train_model_name + '.stagedmodel')

  # selfplay nodes get exported model from training rank
  transfer_file_to_selfplay(state, dcomm, model_path_staged)
  transfer_file_to_selfplay(state, dcomm, eval_model_path)

  if drank > 0:
    # double buffer model write
    # b/c selplay is always running and looking for this new model,
    # a simple f.write crashes due to header/contents size mismatch (loader.cc:165)
    # => we do a staged model transfer. Presence of .minigo denotes that .stagedmodel has
    #    been completely made available.
    # copy staged model to minigo
    copy_file(model_path_staged, model_path, FLAGS.verbose)


def transfer_selfplay_data_to_train(state, dcomm, num_games_per_node, completed_game_paths):
  # only for multinode
  if FLAGS.num_nodes <= 1: return

  # Sync before start
  dcomm.barrier()
  drank  = dcomm.Get_rank()

  logging.info('selfplay data transfer iter %s d-rank %d, num_games_per_node %d', state.iter_num, drank, num_games_per_node)

  # Train gets selfplay data from all other nodes.
  # Each selfplay node waits until it generates num_games == (FLAGS.suggested_games_per_iteration / num_selfplay_nodes)
  # So train rank receives (num_games * num_selfplay_nodes) worth of game files.
  if drank == 0:
    file_bytes = dcomm.gather(None, root=0)
    file_dir = os.path.join(FLAGS.selfplay_dir, state.selfplay_model_name, 'train-node', '17')
    ensure_dir_exists(file_dir, FLAGS.verbose)
    for i in range(FLAGS.num_train_nodes, FLAGS.num_nodes): # write selfplay data from train_nodes through num_nodes
      for j in range(num_games_per_node):
        fileout = os.path.join(file_dir, 'game-data-' + str(j) + '-from-node-' + str(i) + '.tfrecord.zz')
        with open(fileout, 'wb') as f:
          f.write(file_bytes[i][j])
      logging.info('Gather train iter %s from drank %d, wbytes[games 1-%s] %s', state.iter_num, i, num_games_per_node, list(map(len, file_bytes[i])))
  else:
    read_bytes_list = []*num_games_per_node

    # read from only the files that have been completely written.
    game_paths = [ p.replace('completed', 'zz') for p in completed_game_paths ]
    for gamefile in game_paths[:num_games_per_node]:
      with open(gamefile, 'rb') as f:
        read_bytes = f.read()
        read_bytes_list.append(read_bytes)
    file_bytes = dcomm.gather(read_bytes_list, root=0)
    logging.info('Gather selfplay drank %d iter %s rbytes[1-%s] %s', drank, state.iter_num, len(read_bytes_list), list(map(len, read_bytes_list)))


def selfplay(rank):
    local_rank = rank % FLAGS.ranks_per_node
    logging.info('Selfplay on MPI rank %d, local_rank is %d host is %s', rank, local_rank, socket.gethostname())

    try:
        minigo_python.run_concurrent_selfplay(
                      os.path.join(FLAGS.model_dir, '%d.minigo'),
                      FLAGS.cache_size_mb,
                      FLAGS.cache_shards,
                      FLAGS.num_readouts,
                      FLAGS.fastplay_frequency,
                      FLAGS.fastplay_readouts,
                      FLAGS.virtual_losses,
                      FLAGS.dirichlet_alpha,
                      FLAGS.noise_mix,
                      FLAGS.value_init_penalty,
                      FLAGS.target_pruning,
                      FLAGS.policy_softmax_temp,
                      FLAGS.allow_pass,
                      FLAGS.restrict_pass_alive_play_threshold,
                      FLAGS.selfplay_threads,
                      FLAGS.parallel_search,
                      FLAGS.parallel_inference,
                      FLAGS.concurrent_games_per_thread,
                      FLAGS.seed,
                      FLAGS.min_resign_threshold,
                      FLAGS.max_resign_threshold,
                      FLAGS.disable_resign_pct,
                      FLAGS.num_games,
                      FLAGS.run_forever,
                      os.path.join(FLAGS.base_dir, 'abort'),
                      FLAGS.holdout_pct,
                      os.path.join(FLAGS.selfplay_dir, '$MODEL', str(local_rank)),
                      os.path.join(FLAGS.holdout_dir, '$MODEL', str(local_rank)),
                      "",
                      FLAGS.verbose and FLAGS.selfplay_verbose,
                      FLAGS.output_threads,
                      1
                      )
    except Exception:
        logging.error('[Selfplay rank = %d] caught exception.', rank)

def train_complete_cleanup(rank, state, dcomm):
    # create abort file to stop selfplay
    abort_file = os.path.join(FLAGS.base_dir, 'abort')
    logging.info('Creating abort file at %s on g-rank %d node %d', abort_file, rank, rank // FLAGS.ranks_per_node)
    ensure_file_exists(abort_file, FLAGS.verbose)

    # multinode eval, sync train_times.txt
    filename = os.path.join(FLAGS.model_dir, 'train_times.txt')
    transfer_file_to_selfplay(state, dcomm, filename)


def load_train_times():
  models = []
  path = os.path.join(FLAGS.model_dir, 'train_times.txt')
  with tf.io.gfile.GFile(path, 'r') as f:
    for line in f.readlines():
      line = line.strip()
      if line:
        timestamp, name = line.split(' ')
        path = os.path.join(FLAGS.model_dir, name + '.evalfp32minigo')
        models.append((float(timestamp), name, path))
  return models


def eval_models(rank, comm, eval_ranks):
    # explicitly switch off FP16 AMP signals
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '0'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '0'

    # split comm into num_nodes
    split_size       = len(eval_ranks) // FLAGS.num_nodes
    node_index       = rank // FLAGS.ranks_per_node
    start_index      = split_size * node_index
    my_eval_ranks    = eval_ranks[ start_index : start_index + split_size ]
    ecomm  = get_group_comm(comm, my_eval_ranks)
    erank  = ecomm.Get_rank()
    esize  = ecomm.Get_size()

    # group of all leaders in eranks
    leaders = eval_ranks[::split_size]
    legroup = comm.group.Incl(leaders)
    lecomm = comm.Create_group(legroup)

    # args/Flags
    assert FLAGS.num_eval_games % esize == 0, "Number of eval workers should divide FLAGS.num_eval_games evenly"
    num_games = math.ceil(FLAGS.num_eval_games // esize)

    # load models
    models = load_train_times()
    models_per_node = 1 + len(models) // FLAGS.num_nodes
    model_start_index = node_index * models_per_node
    mymodels = models[ model_start_index : model_start_index + models_per_node ]

    # global win rate
    epochwinrates = np.zeros(models_per_node, dtype=np.float32)

    for i, (timestamp, name, eval_model_path) in enumerate(mymodels):
        try:
            logging.info('erank %d of (%d) node-index %d running eval-model %s', erank, esize, node_index, eval_model_path)
            win_rate = minigo_python.run_eval(
                          FLAGS.resign_enabled,
                          FLAGS.resign_threshold,
                          FLAGS.seed,
                          FLAGS.virtual_losses,
                          FLAGS.value_init_penalty,
                          eval_model_path,
                          "",
                          FLAGS.num_eval_readouts,
                          FLAGS.target_path,
                          "",
                          FLAGS.num_eval_readouts,
                          num_games,
                          FLAGS.verbose and FLAGS.selfplay_verbose
                          )

            mwinrate = np.array(win_rate, dtype=np.float32)
            winrates = np.zeros(esize, dtype=np.float32)
            ecomm.Gather(mwinrate, winrates, root=0)

            if erank == 0:
               net_win_rate = sum(winrates) / (esize * 1.0)
               epochwinrates[i] = net_win_rate
               logging.info('erank %d of (%d) win_rate %3.5f net_win_rate for model %s is %3.5f', erank, esize, win_rate, name, net_win_rate)

            ecomm.Barrier()

        except Exception:
            logging.error('[Eval erank = %d] caught exception.', erank)

    status = 'aborted', None
    # accumulate multinode eval
    if rank in leaders:
        lerank  = lecomm.Get_rank()
        lesize  = lecomm.Get_size()
        logging.info('node-index %d rank %d erank %d of %d lerank %d of %d',node_index, rank, erank, esize, lerank, lesize)

        lecomm.Barrier()
        combine_epoch_winrates = None if lerank else np.zeros(models_per_node*FLAGS.num_nodes, dtype=np.float32)
        lecomm.Gather(epochwinrates, combine_epoch_winrates, root=0)

        if lerank == 0:
            iter_evaluated = 0
            accuracy_evaluated = 0
            timestamp_to_log = 0
            logging.info('combined epoch win rates is {}'.format(combine_epoch_winrates))
            for i, (timestamp_seconds, name, eval_model_path) in enumerate(models):
               log_start(key=constants.EVAL_START, metadata={'epoch_num': i + 1})
               log_event(key=constants.EVAL_ACCURACY, value=float(combine_epoch_winrates[i]), metadata={'epoch_num': i + 1})
               log_end(key=constants.EVAL_STOP, metadata={'epoch_num': i + 1})

               iter_evaluated += 1
               accuracy_evaluated = combine_epoch_winrates[i]
               timestamp_to_log = timestamp_seconds
               if combine_epoch_winrates[i] >= FLAGS.winrate:
                  print('Model {} beat target after {}s'.format(name, timestamp_seconds))
                  status = 'success', timestamp_seconds
                  break

            log_event(key='eval_result', value=float(accuracy_evaluated), metadata={'epoch_num': iter_evaluated, 'timestamp' : timestamp_to_log})

        lecomm.Barrier()

    # sync all ranks on this node
    ecomm.Barrier()
    return status


def distribute_mpiranks(rank, size):
    # Distribute MPI ranks between train, selfplay, and eval
    # For example with a DGX2, ranks_per_node = 17, 16 gpu work related, and 1 for data-transfer
    #                          -on node-0- rank-16 is data-transfer, ranks-0 through FLAGS.num_gpus_train is train, rest are selfplay,
    #                          -on node-1- rank-16 is data-transfer, ranks-0-15 are selfplay
    #                                   .
    #                                   .
    #                          -on node-n- rank-16 is data-transfer, rank-0-15 are selfplay
    assert FLAGS.ranks_per_node > 1, "need >1 ranks/node as 1 is reserved for data transfer"

    #
    gpus_per_node = (FLAGS.ranks_per_node - 1) // FLAGS.procs_per_gpu

    if FLAGS.num_gpus_train > gpus_per_node:
        # when training on >1 node, we use all gpus of a node for train.
        train_nodes   = max(1, FLAGS.num_gpus_train // gpus_per_node)
        train_ranks   = [r for r in list(range(0, FLAGS.ranks_per_node * train_nodes )) if r % FLAGS.ranks_per_node < gpus_per_node ]
        n_train_ranks = [r for r in list(range(0, FLAGS.ranks_per_node * train_nodes )) if r % FLAGS.ranks_per_node >= gpus_per_node ]
        assert FLAGS.num_gpus_train == len(train_ranks), "for multinode training, num_gpus_train should use all gpus."
    else:
        train_ranks   = list(range(0, FLAGS.num_gpus_train))
        n_train_ranks = [r for r in list(range(0, FLAGS.ranks_per_node)) if r % gpus_per_node <= train_ranks[-1]]
        n_train_ranks = [0]

    eval_ranks = [r for r in list(range(0, size)) if r % FLAGS.ranks_per_node < gpus_per_node ]
    data_transfer_ranks = list(range(FLAGS.ranks_per_node-1, size, FLAGS.ranks_per_node))
    selfplay_ranks = [r for r in list(range(0, size)) if r not in (train_ranks + n_train_ranks + data_transfer_ranks)]

    # multi-gpu training can be multinode as well.
    node_of_first_selfplay_rank = selfplay_ranks[0] // FLAGS.ranks_per_node
    selfplay_data_transfer_ranks = data_transfer_ranks[node_of_first_selfplay_rank:]


    assert train_ranks,          "train ranks is not populated."
    assert data_transfer_ranks,  "data_transfer ranks ranks is not populated."
    assert eval_ranks,           "eval ranks is not populated."
    assert selfplay_ranks,       "selfplay ranks is not populated."

    #index to bind to gpus
    FLAGS.rank_gpu_index = ((rank % FLAGS.ranks_per_node) % gpus_per_node)

    # bind ranks to GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.rank_gpu_index)

    # for multinode training, tzranks is the set of ranks that
    # participate in input data sync before train can start.
    if rank in train_ranks:
        tzranks = train_ranks[FLAGS.rank_gpu_index:][::gpus_per_node]
    else:
        tzranks = [0]

    if rank == 0:
        logging.info('train ranks are %s', train_ranks)
        logging.info('train node leader ranks are %s', tzranks)
        logging.info('data_transfer ranks are %s', data_transfer_ranks)
        logging.info('eval ranks are %s', eval_ranks)
        logging.info('selfplay ranks are %s', selfplay_ranks)
        logging.info('selfplay data_transfer ranks are %s', selfplay_data_transfer_ranks)

    return (train_ranks, tzranks, data_transfer_ranks, eval_ranks, selfplay_ranks, selfplay_data_transfer_ranks)

def get_group_comm(comm, ranks):
    # Create a grouped mpi communicator with the ranks
    assert len(ranks) > 0, "cannot create group as ranks is empty"
    xgroup = comm.group.Incl(ranks)
    xcomm  = comm.Create_group(xgroup)

    return xcomm

def init_logger():
    #
    logger = logging.getLogger()
    if not FLAGS.verbose:
        logger.setLevel(logging.ERROR)
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                      '%Y-%m-%d %H:%M:%S')
        for handler in logger.handlers:
            handler.setFormatter(formatter)


def init_flags(rank):
    # initialize all flags related to dirs.
    FLAGS.base_dir = os.path.join(FLAGS.base_dir + '-node-' + str(rank // FLAGS.ranks_per_node))
    FLAGS.flags_dir = os.path.join(FLAGS.base_dir, 'flags')
    FLAGS.work_dir = os.path.join(FLAGS.base_dir, 'work_dir')
    FLAGS.model_dir = os.path.join(FLAGS.base_dir, 'models')
    FLAGS.selfplay_log_dir = os.path.join(FLAGS.base_dir, 'logs')
    FLAGS.golden_chunk_dir = os.path.join(FLAGS.base_dir, 'data/golden_chunks')
    FLAGS.holdout_dir = os.path.join(FLAGS.base_dir, 'data/holdout')
    FLAGS.selfplay_dir = os.path.join(FLAGS.base_dir, 'data/selfplay')

def init_from_checkpoint(rank, comm):
    # run checkpoint init
    if rank % FLAGS.ranks_per_node == 0:
        logging.info('Wiping dir %s', FLAGS.base_dir)
        remove_tree(FLAGS.base_dir, FLAGS.verbose)
        dirs = [FLAGS.model_dir, FLAGS.selfplay_dir, FLAGS.holdout_dir, FLAGS.golden_chunk_dir,
                FLAGS.work_dir, FLAGS.selfplay_log_dir, FLAGS.flags_dir ]

        for d in dirs:
          ensure_dir_exists(d, FLAGS.verbose);

        # Copy the checkpoint data to the correct location.
        copy_tree(os.path.join(FLAGS.checkpoint_dir, 'data/selfplay'), FLAGS.selfplay_dir, FLAGS.verbose)
        copy_tree(os.path.join(FLAGS.checkpoint_dir, 'work_dir'), FLAGS.work_dir, FLAGS.verbose)
        copy_tree(os.path.join('ml_perf/flags', FLAGS.board_size), FLAGS.flags_dir, FLAGS.verbose)

    # sync
    comm.barrier()

    # pre-process checkpoint files. No need b/c we filter at sample-records.
    # sample_checkpoint_examples(rank)


def init_state(rank):
    model_dirs = list_selfplay_dirs(FLAGS.selfplay_dir)
    assert model_dirs, "checkpoint copy failure, model_dirs is empty"
    model_num = int(os.path.basename(model_dirs[0]))
    state = State(model_num)
    logging.info('rank %d, model_num from checkpoint is %s', rank, model_num)

    return state, model_num

def mlperf_log_hyperparams(rank):
    if rank == 0:
        log_event(key='lr_rates', value=FLAGS.lr_rates, sync=False)
        log_event(key='lr_boundaries', value=FLAGS.lr_boundaries, sync=False)
        log_event(key='train_batch_size', value=FLAGS.train_batch_size, sync=False)
        log_event(key='virtual_losses', value=FLAGS.virtual_losses, sync=False)
        log_event(key=constants.SEED, value=FLAGS.seed, sync=False)
        log_event(key=constants.OPT_WEIGHT_DECAY, value=FLAGS.l2_strength, sync=False)
        log_event(key='filter_amount', value=FLAGS.train_filter, sync=False)
        log_event(key='num_readouts', value=FLAGS.num_readouts, sync=False)
        log_event(key='value_init_penalty', value=FLAGS.value_init_penalty, sync=False)
        log_event(key='holdout_pct', value=FLAGS.holdout_pct, sync=False)
        log_event(key='disable_resign_pct', value=FLAGS.disable_resign_pct, sync=False)
        log_event(key='resign_threshold', value=FLAGS.resign_threshold, sync=False)
        log_event(key='virtual_losses', value=FLAGS.virtual_losses, sync=False)
        log_event(key='min_selfplay_games_per_generation', value=FLAGS.min_games_per_iteration)
        log_event(key='value_init_penalty', value=FLAGS.value_init_penalty, sync=False)
        log_event(key='window_size', value=FLAGS.window_size, sync=False)
        log_event(key='eval_games', value=FLAGS.num_eval_games, sync=False)

def mlperf_log_init_stop_run_start(rank, comm):
    if rank == 0: log_end(key=constants.INIT_STOP)
    comm.barrier()
    if rank == 0: log_start(key=constants.RUN_START)
    comm.barrier()


def main(unused_argv):
    """Run the reinforcement learning loop."""

    # logger
    init_logger()

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    node = rank // FLAGS.ranks_per_node
    logging.info('[MPI Init] rank %d of (%d) host is %s', rank, size, socket.gethostname())

    # distribute ranks
    # tzranks is the set of training ranks used to sync training input.
    train_ranks, tzranks, data_transfer_ranks, eval_ranks, \
    selfplay_ranks, selfplay_data_transfer_ranks = distribute_mpiranks(rank, size)

    # create grouped mpi comm for train/data-transfer ranks
    tcomm  = get_group_comm(comm, train_ranks)
    tzcomm = get_group_comm(comm, tzranks)
    tdcomm = get_group_comm(comm, train_ranks + data_transfer_ranks)
    dcomm  = get_group_comm(comm, data_transfer_ranks)

    # calculate node counts, and selfplay games/node.
    FLAGS.num_nodes          = size // FLAGS.ranks_per_node
    FLAGS.num_train_nodes    = max(1, len(train_ranks) * FLAGS.procs_per_gpu // (FLAGS.ranks_per_node - 1))
    FLAGS.num_selfplay_nodes = max(1, len(selfplay_data_transfer_ranks))
    num_games_per_node       = math.ceil(FLAGS.suggested_games_per_iteration / FLAGS.num_selfplay_nodes)

    #
    # init_start is once per worker
    log_start(key=constants.INIT_START)

    # set flags
    init_flags(rank)

    # init checkpoint state
    init_from_checkpoint(rank, comm)

    # Initialize state off selfplay data from the checkpoint.
    state, checkpoint_model_num = init_state(rank)

    # print mlperf hyperparams
    mlperf_log_hyperparams(rank)

    # init hvd for training
    if rank in train_ranks:
        tstate = minigo_train.init_train(rank, tcomm, FLAGS.model_dir)

    # mlperf init stops, run starts
    # print once per run
    mlperf_log_init_stop_run_start(rank, comm)

    # train and orchestrate data-transfer b/w train, and selfplay nodes.
    if rank in (train_ranks + data_transfer_ranks):

        # restart timer since we are now training from here
        state.start_time = time.time()

        while state.iter_num <= FLAGS.iterations:
            state.iter_num += 1

            # epoch start
            if rank == 0: log_start(key=constants.EPOCH_START, metadata={'epoch_num': state.iter_num})

            # get selfplay data. Skip first iteration
            if rank in data_transfer_ranks and state.iter_num > 1 + checkpoint_model_num:
                # wait until each selfplay generates enough data
                completed_game_paths = []
                if rank in selfplay_data_transfer_ranks:
                    with logged_timer('[Selfplay data generation node {}] iter {} time'.format(node, state.iter_num)):
                        completed_game_paths = wait_for_training_examples(state, num_games_per_node, '.completed')

                #
                dcomm.barrier()

                # transfer all selfplay to train node
                with logged_timer('[Selfplay data transfer node {}] iter {} time'.format(node, state.iter_num)):
                    transfer_selfplay_data_to_train(state, dcomm, num_games_per_node, completed_game_paths)

            #
            tdcomm.barrier()

            # train
            if rank in train_ranks:
                with logged_timer('[Train rank {}] Iteration {} time'.format(rank, state.iter_num)):
                    run_train(state, rank, tstate, tcomm, tzcomm)

            #
            tdcomm.barrier()

            # transfer trained model
            if rank in data_transfer_ranks:
                with logged_timer('[Model transfer node {}] iter {} time'.format(node, state.iter_num)):
                     transfer_trained_model_to_selfplay(state, dcomm)

            #
            tdcomm.barrier()

            # epoch stop
            if rank == 0: log_end(key=constants.EPOCH_STOP, metadata={'epoch_num': state.iter_num})

        # done with training. stop selfplay and prepare for eval.
        if rank in data_transfer_ranks:
            train_complete_cleanup(rank, state, dcomm)

    # Run selfplay in while(1) loop; ends when abort file is detected
    if rank in selfplay_ranks:
        logging.info('rank %d running selfplay', rank)
        selfplay(rank)

    logging.info('[Train/Selfplay done] rank %d of (%d) host is %s', rank, size, socket.gethostname())
    comm.barrier()

    # eval
    status = None, None
    if rank in eval_ranks:
        status = eval_models(rank, comm, eval_ranks)

    # done
    comm.barrier()

    # shutdown hvd for training
    if rank in train_ranks:
        minigo_train.stop_train(rank)

    # log status from eval
    if rank == 0:
        result, timestamp_seconds = status
        mlperf_timestamp_ms = int((state.start_time + timestamp_seconds) * 1000) if timestamp_seconds is not None else None
        log_end(key=constants.RUN_STOP, time_ms=mlperf_timestamp_ms, metadata={constants.STATUS: result})

    #
    comm.barrier()
    logging.info('[MPI done] rank %d of (%d) host is %s', rank, size, socket.gethostname())


if __name__ == '__main__':
    app.run(main)
