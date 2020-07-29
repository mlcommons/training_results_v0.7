import collections
import os
import subprocess
import numpy as np
from mlperf_logging.mllog import constants as mlperf_constants
from mlperf_logging import mllog


mllogger = mllog.get_mllogger()

def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)
def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)
def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)
def _log_print(logger, *args, **kwargs):
    sync = kwargs.pop('sync', False)
    uniq = kwargs.pop('uniq', True)

    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 3
    if 'value' not in kwargs:
        kwargs['value'] = None

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

