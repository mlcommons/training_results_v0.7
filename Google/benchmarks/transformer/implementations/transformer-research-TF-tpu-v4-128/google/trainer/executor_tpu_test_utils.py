# Lint as: python2, python3
"""Common utils shared by various tpu tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import base_trial
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import compat as tf
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import executor
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import model_registry
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import cluster_factory
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import py_utils
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


def _CopyCfg(cfg, jobname):
  p = cfg.Copy()
  p.cluster.job = jobname
  return p


class BaseExecutorTest(test_utils.TestCase):
  """Base class for the test cases."""

  def __init__(self, *args, **kwargs):
    super(BaseExecutorTest, self).__init__(*args, **kwargs)
    self._trial = base_trial.NoOpTrial()

  def _GetExecutorParams(self, model_name, replacement_input_file_pattern=None):
    """Retrieve the params needed to instantiate an Executor for unit tests.

    Args:
     model_name: Name of the model to instantiate.
     replacement_input_file_pattern: A test file pattern to use.

    Returns:
     - ps_params_dict: High-level task_name -> ProgramScheduleParams
     - train_cfg: Either a SingleTaskModelParams or MultiTaskModelParams
    """
    cluster = cluster_factory.Current()
    cluster.params.job = 'executor_tpu'
    cluster.params.mode = 'sync'
    cluster.params.task = 0
    cluster.params.worker.name = '/job:localhost'
    cluster.params.worker.replicas = 1
    cluster.params.worker.gpus_per_replica = 0
    cluster.params.evaler.name = '/job:localhost'
    cluster.params.evaler.replicas = 1
    cluster.params.evaler.gpus_per_replica = 0
    cluster.params.decoder.name = '/job:localhost'
    cluster.params.decoder.replicas = 1
    cluster.params.decoder.gpus_per_replica = 0
    cluster.params.ps.name = '/job:localhost'
    cluster.params.ps.replicas = 1
    cluster.params.ps.gpus_per_replica = 0
    cluster.params.worker.tpus_per_replica = 2
    cluster.params.worker.num_tpu_hosts = 1
    cluster.params.worker.replicas = 1
    cluster.params.worker.devices_per_split = 1
    ps_params_dict, train_cfg = executor.GetExecutorParams(
        model_name, cluster.params, model_registry)
    if replacement_input_file_pattern:
      if len(ps_params_dict) > 1:
        # Multi-task case, need to fix all the input_params.
        for task_str, input_params in train_cfg.input.IterParams():
          input_params.file_pattern = replacement_input_file_pattern

        for task_str, ps_params in ps_params_dict.items():
          for unused_dataset, params in ps_params.task_dict.items():
            params.input = train_cfg.input.Get(task_str)
      else:
        train_cfg.input.file_pattern = replacement_input_file_pattern
        for unused_task_str, ps_params in ps_params_dict.items():
          for unused_dataset, params in ps_params.task_dict.items():
            params.input = train_cfg.input

    return ps_params_dict, train_cfg

  @flagsaver.flagsaver
  def _testExecutorTpuHelper(self,
                             cfg,
                             ps_params_dict,
                             metric_name='log_pplx',
                             enqueue_max_steps=20,
                             eval_samples_per_summary=4,
                             decoder_samples_per_summary=0):
    if hasattr(cfg.input, 'file_pattern'):
      # Unit tests should not read data from prod.
      self.assertNotIn('/REDACTED/', cfg.input.file_pattern)
      self.assertNotIn('/placer/', cfg.input.file_pattern)

    FLAGS.enable_asserts = True
    FLAGS.tpu_compatible = True
    FLAGS.xla_device = 'tpu'
    FLAGS.enable_asserts = False
    tf_master = 'local'
    tf.Session.reset('local')

    py_utils.ClearTpuDevice()
    logdir = self.create_tempdir().full_path
    cfg.cluster.worker.tpus_per_replica = 2
    cfg.cluster.worker.num_tpu_hosts = 1
    cfg.cluster.worker.replicas = 1
    cfg.cluster.worker.devices_per_split = 1

    tpu_executor = executor.ExecutorTpu(
        cfg, ps_params_dict, '', logdir, tf_master, trial=self._trial)

    tpu_executor.Start()

    return tpu_executor
