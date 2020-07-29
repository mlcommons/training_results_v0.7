import collections
import os
import subprocess
import numpy as np
from mlperf_logging.mllog import constants
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
mllogger = mllog.get_mllogger()

def all_reduce(v):
    return mpiwrapper.allreduce(v)

def _mx_resnet_print(logger, key, 
                     val=None, metadata=None, namespace=None, 
                     stack_offset=3, uniq=True):
    rank = mpiwrapper.rank()
    if (uniq and rank == 0) or (not uniq):
        logger(key=key, value=val, metadata=metadata,
               stack_offset=stack_offset)

def mx_resnet_print_start(*args, **kwargs):
    _mx_resnet_print(mllogger.start, *args, **kwargs)

def mx_resnet_print_end(*args, **kwargs):
    _mx_resnet_print(mllogger.end, *args, **kwargs)

def mx_resnet_print_event(*args, **kwargs):
    _mx_resnet_print(mllogger.event, *args, **kwargs)


def mlperf_submission_log(benchmark):

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

    mx_resnet_print_event(
        key=constants.SUBMISSION_BENCHMARK,
        val=benchmark,
        )

    mx_resnet_print_event(
        key=constants.SUBMISSION_ORG,
        val='Dell EMC')

    mx_resnet_print_event(
        key=constants.SUBMISSION_DIVISION,
        val='closed')

    mx_resnet_print_event(
        key=constants.SUBMISSION_STATUS,
        val='onprem')

    mx_resnet_print_event(
        key=constants.SUBMISSION_PLATFORM,
        val='{}xSUBMISSION_PLATFORM_PLACEHOLDER'.format(num_nodes))

def resnet_max_pool_log(input_shape, stride):
    downsample = 2 if stride == 2 else 1
    output_shape = (input_shape[0], 
                    int(input_shape[1]/downsample), 
                    int(input_shape[2]/downsample))

    return output_shape


def resnet_begin_block_log(input_shape):
    return input_shape


def resnet_end_block_log(input_shape):
    return input_shape


def resnet_projection_log(input_shape, output_shape):
    return output_shape


def resnet_conv2d_log(input_shape, stride, out_channels, bias):
    downsample = 2 if (stride == 2 or stride == (2, 2)) else 1
    output_shape = (out_channels, 
                    int(input_shape[1]/downsample), 
                    int(input_shape[2]/downsample))

    return output_shape


def resnet_relu_log(input_shape):
    return input_shape


def resnet_dense_log(input_shape, out_features):
    shape = (out_features)
    return shape


def resnet_batchnorm_log(shape, momentum, eps, center=True, scale=True, training=True):
    return shape
