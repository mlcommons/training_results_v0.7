import collections
import os
import subprocess
import numpy as np
from mlperf_logging.mllog import constants as mlperf_constants
from mlperf_logging import mllog

class MPIWrapper(object):
    def __init__(self):
        self.comm = None
        self.MPI = None

    def get_comm(self):
        if self.comm is None:
            import mpi4py
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI = MPI

        return self.comm

    def barrier(self):
        c = self.get_comm()
        # NOTE: MPI_Barrier is *not* working reliably at scale. Using MPI_Allreduce instead.
        #c.Barrier() 
        val = np.ones(1, dtype=np.int32)
        result = np.zeros(1, dtype=np.int32)
        c.Allreduce(val, result)

    def allreduce(self, x):
        c = self.get_comm()
        rank = c.Get_rank()
        val = np.array(x, dtype=np.int32)
        result = np.zeros_like(val, dtype=np.int32)
        c.Allreduce([val, self.MPI.INT], [result, self.MPI.INT]) #, op=self.MPI.SUM)
        return result

    def rank(self):
        c = self.get_comm()
        return c.Get_rank()

mpiwrapper=MPIWrapper()

def all_reduce(v):
    return mpiwrapper.allreduce(v)

mllogger = mllog.get_mllogger()

def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)
def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)
def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)
def _log_print(logger, *args, **kwargs):
    rank = mpiwrapper.rank()
    uniq = kwargs.pop('uniq', True)

    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 3
    if 'value' not in kwargs:
        kwargs['value'] = None

    if (uniq and rank == 0) or (not uniq):
        logger(*args, **kwargs)

    return

def mlperf_submission_log(benchmark):

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

    log_event(
        key=mlperf_constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    log_event(
        key=mlperf_constants.SUBMISSION_ORG,
        value='NVIDIA')

    log_event(
        key=mlperf_constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        key=mlperf_constants.SUBMISSION_STATUS,
        value='onprem')

    log_event(
        key=mlperf_constants.SUBMISSION_PLATFORM,
        value='{}xSUBMISSION_PLATFORM_PLACEHOLDER'.format(num_nodes))
