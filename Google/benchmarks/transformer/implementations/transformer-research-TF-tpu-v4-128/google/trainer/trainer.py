# Lint as: python2, python3
r"""Trainer for babelfish models.

You should follow {allocator_,}trainer{,_sync}.REDACTED's instruction when
training a model on REDACTED.

You can also run the trainer locally on your workstation:
$ REDACTED build -c opt --define=babelfish_task=speech \
    learning/REDACTED/research/babelfish/trainer/trainer
$ REDACTED-py3/bin/learning/REDACTED/research/babelfish/trainer/trainer \
   --logtostderr \
   --model=speech.wsj.WSJ \
   --logdir=$HOME/tmp/wsj \
   --REDACTED_port=12345 \
   --run_locally=cpu \
   --mode=sync

If you have a GPU, add --config=cuda to the build command and pass
'gpu' to --run_locally.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading

from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import compat as tf
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import model_imports
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import trainer as lingvo_trainer
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import checkpointer
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import cluster as lingvo_cluster
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import cluster_factory
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import py_utils
from REDACTED.learning.REDACTED.research.babelfish import REDACTED_utils
from REDACTED import httpserver2


# TODO(jonathanasdf): Remove and don't write inference graph by default.
tf.flags.DEFINE_bool('write_inference_graph', True,
                     'Whether to write inference graph.')

tf.flags.DEFINE_string('guzzler_server_address', None,
                       'The server address, if using Data Guzzler.')

tf.flags.DEFINE_integer(
    'guzzler_timeout_ms', 600000,
    'The amount of time the DataGuzzler servers have '
    'to respond before an error is thrown.')

tf.flags.DEFINE_string(
    'precompute_cache_path', None,
    'The path to where files containing precomputed '
    'preprocessed inputs can be found.  Typically used '
    'for precomputing Tensors for evaluation datasets to '
    'speed up decoding.')

tf.flags.DEFINE_boolean('use_custom_saver', False,
                        'Uses customized saver if True.')

tf.flags.DEFINE_boolean(
    'use_lingvo_cluster', False, 'Uses lingvo_cluster._Cluster instead of '
    'babelfish.cluster._Cluster.')

# in REDACTED config set args { pythreadz_port = '%port_pythreadz%' }
tf.flags.DEFINE_integer('pythreadz_port', 0, 'optional port for /pythreadz')

FLAGS = tf.flags.FLAGS




class RunnerManager(lingvo_trainer.RunnerManager):
  """Helper class for managing runners."""

  # Override some modules.
  #inference_graph_exporter = inference_graph_exporter
  #model_registry = model_registry

  # Override the Controller implementation from lingvo with our subclass.
  # pylint: disable=invalid-name
  #Controller = Controller

  # pylint: enable=invalid-name

  def MaybeLaunchTensorFlow(self):
    """Starts TF machinary in this process."""
    if FLAGS.REDACTED_run_locally or FLAGS.run_locally:
      return

    tf.logging.info('Launching tensorflow.')
    #monitor.OutputMetrics(presence=True)

    # Take control of the status message, otherwise the tf server will on the
    # following line.
    REDACTED_utils.SetBorgStatusMessage('Starting up...')

    # Launch the in-proc tensorflow server.
    REDACTED_utils.LaunchTensorFlowServer()

    target = FLAGS.tf_master
    if not target.startswith('localhost'):
      # E.g., train_client is configured w/ FLAGS.tf_master pointing to
      # another job. In that case, uses 'local' so that we start a
      # server listening to the --REDACTED_port.
      target = 'local'
    with tf.Session(target).as_default():
      value = (tf.constant(1.) + tf.constant(1.)).eval()
    assert value == 2.0, 'Something is really wrong.'
    tf.logging.info('Launched tensorflow.')

  def MaybeConfigRunLocally(self):
    """Update flags if configured to run locally."""
    if not FLAGS.run_locally:
      # Do nothing
      return

    super(RunnerManager, self).MaybeConfigRunLocally()
    FLAGS.tf_master = 'local'
    FLAGS.controller_job = '/job:localhost'
    FLAGS.worker_job = '/job:localhost'
    FLAGS.ps_job = '/job:localhost'
    FLAGS.input_job = '/job:localhost'
    FLAGS.evaler_job = '/job:localhost'
    FLAGS.decoder_job = '/job:localhost'

    # Some flags convenient to have
    FLAGS.flagz_disabled_flags = ''
    FLAGS.REDACTED_use_bfloat16_for_sendrecv = False
    FLAGS.REDACTED_session_gc_seconds = 86400
    FLAGS.REDACTED_use_gpuprof = False
    FLAGS.REDACTED_timeline_step = 100
    FLAGS.REDACTED_collect_cpu_allocator_stats = True

  def Start(self):
    if FLAGS.pythreadz_port:
      http_server = httpserver2.Builder(
          FLAGS.pythreadz_port, 'pythreadz_server', num_threads=10).Build()
      http_server.AddRedirect('/', '/pythreadz')
      t = threading.Thread(target=http_server.Serve)
      t.daemon = True
      t.start()
    if FLAGS.use_lingvo_cluster:
      # TODO(zhifengc): simplify VarPlacer, potentially use sess.list_devices().
      #
      # babelfish.cluster._Cluster._MakeDeviceString overrides the way we
      # construct CPU device string for variable placement, with
      # task_id \in {0 ... ps.num_replicas - 1}, e.g. for ps replica 12
      #   /job:trainer/replica:12/task:0/device:CPU:0
      # For TPU training and sharded variable placement on different TPU hosts
      # we need to construct strings like
      #   /job:trainer/replica:0/task:12/device:CPU:0
      # instead.
      cluster_factory.SetCluster(lingvo_cluster._Cluster)  # pylint: disable=protected-access
    # Legacy code for automatic inference graph writing.
    # TODO(jonathanasdf): Remove and don't write inference graph by default.
    if (FLAGS.write_inference_graph and
        (not FLAGS.job or 'controller' in FLAGS.job) and
        FLAGS.mode in ('sync', 'async')):
      self.WriteInferenceGraph()
    super(RunnerManager, self).Start()

  def UpdateClusterParamsFromFlags(self, cluster, job_name):
    super(RunnerManager, self).UpdateClusterParamsFromFlags(cluster, job_name)
    if FLAGS.guzzler_server_address is not None:
      cluster.guzzler_server_address = FLAGS.guzzler_server_address
      cluster.guzzler_timeout_ms = FLAGS.guzzler_timeout_ms
      cluster.guzzler_graph_dir = os.path.join(FLAGS.logdir, 'train')
    if FLAGS.precompute_cache_path:
      cluster.precompute_cache_path = FLAGS.precompute_cache_path


def main(unused_argv):
  RunnerManager(FLAGS.model).Start()


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('model')
  FLAGS(sys.argv, known_only=True)
  model_imports.ImportParams(FLAGS.model)
  FLAGS.unparse_flags()
  tf.app.run(main)
