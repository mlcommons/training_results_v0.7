# Lint as: python2, python3
"""Specification of a training cluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import cluster as lingvo_cluster
import numpy as np
from six.moves import range


class _Cluster(lingvo_cluster._Cluster):  # pylint: disable=protected-access
  """The whole training cluster from a single task's point of view."""

  @classmethod
  def _JobSpec(cls, replicas):
    p = super(_Cluster, cls)._JobSpec(replicas)
    p.Define('spus_per_replica', 0,
             'The number of spu cores to use per replica.')
    return p

  @classmethod
  def _MakeDeviceString(cls, job_name, task_id, device_name, device_id):
    # In REDACTED, we use replica, not task.
    return '%s/replica:%d/task:0/device:%s:%d' % (job_name, task_id,
                                                  device_name, device_id)

  @classmethod
  def Params(cls):
    """Defaults parameters for a cluster."""
    p = super(_Cluster, cls).Params()
    p.Define('inference_client', cls._JobSpec(1),
             'The inference graph generator job.')
    p.Define('guzzler_server_address', None,
             'The address of the data guzzler server pool, if used.')
    p.Define(
        'guzzler_timeout_ms', 600000,
        'The amount of time the guzzler servers have '
        'to respond before an error is thrown.')
    p.Define('guzzler_graph_dir', None,
             'The directory to publish the guzzler Dataset graph to.')
    p.Define(
        'precompute_cache_path', None,
        'The path to the files containing precomputed preprocessed inputs.')
    return p

  def __init__(self, params):
    self._params = params.Copy()
    p = self.params

    if p.job == 'inference_client':
      assert p.inference_client.replicas >= 1
      assert p.inference_client.tpus_per_replica >= 0
      self._job_spec = p.inference_client
    else:
      super(_Cluster, self).__init__(params)

  @property
  def spus_per_replica(self):
    return self._job_spec.spus_per_replica

  @property
  def guzzler_server_address(self):
    return self.params.guzzler_server_address

  @property
  def guzzler_timeout_ms(self):
    return self.params.guzzler_timeout_ms

  @property
  def guzzler_graph_dir(self):
    return self.params.guzzler_graph_dir

  @property
  def precompute_cache_path(self):
    return self.params.precompute_cache_path

  @property
  def input_targets(self):
    """Returns a list of network addresses of the input job.

    Typically, p.targets is either a BNS job prefix, or a list
    of comma-separated network addresses (host:port, ip:port, or
    grpc://) list.
    """
    p = self.params.input
    if p.targets.startswith('/bns') and (',' not in p.targets):
      # We assume it's a bns job prefix.
      return ['{}/{}'.format(p.targets, i) for i in range(p.replicas)]
    else:
      # Otherwise, it's typically a list of comma-separated network addresses.
      return super(_Cluster, self).input_targets

  @property
  def available_devices(self):
    """Returns all compute devices available in a 2D array.

    Returns:
      A 2D array (python list of python lists) of strings. ret[i, j]
      is the j-th visible device on i-th visible replica.
    """
    if self.job == 'inference_client':
      ret = np.empty((1, self.num_devices_per_split), np.object)
      for i in range(self.num_devices_per_split):
        ret[0, i] = '/device:TPU:%d' % i
      return ret

    return super(_Cluster, self).available_devices

  def GetPlacer(self, strategy=None):
    """Returns a device function for placing ops within the cluster.

    Args:
      strategy: A string. Identifier for a placement strategy. By default,
        we use a least loaded policy to place variables.

    Returns:
      Returns a device function can be used in tf.device().

    Raises:
      ValueError: when strategy is not supported.
    """
    if self.job == 'inference_client':
      return _InferenceSingleCorePlacer(self).DeviceFunction
    return super(_Cluster, self).GetPlacer(strategy)


# TODO(rohananil): Extend this for placing model explicitly to different cores.
class _InferenceSingleCorePlacer(lingvo_cluster.VarPlacer):
  """Placer a variable on core 0 of TPU for inference.
  """

  def _AssignVar(self, var_op):
    del var_op
    return '/device:TPU:0'
