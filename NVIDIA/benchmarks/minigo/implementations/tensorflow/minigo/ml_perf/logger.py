import numpy as np

from mlperf_logging import mllog


mllogger = mllog.get_mllogger()

def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)
def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)
def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)
def _log_print(logger, *args, **kwargs):
    rank = mpiwrapper.rank()
    sync = kwargs.pop('sync', False)
    uniq = kwargs.pop('uniq', True)
    if sync:
        mpiwrapper.barrier()

    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 1
    if 'value' not in kwargs:
        kwargs['value'] = None

    if (uniq and rank == 0) or (not uniq):
        logger(*args, **kwargs)

    if sync:
        mpiwrapper.barrier()

    return


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

