import collections
import os
import subprocess
import torch

from mlperf_logging.mllog import constants
from seq2seq.utils import configure_logger, log_event


def mlperf_submission_log(benchmark):
    num_nodes = os.environ.get('SLURM_NNODES', 1)
    if int(num_nodes) > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    configure_logger(benchmark)

    log_event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    log_event(
        key=constants.SUBMISSION_ORG,
        value='Fujitsu')

    log_event(
        key=constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        key=constants.SUBMISSION_STATUS,
        value='onprem')

    log_event(
        key=constants.SUBMISSION_PLATFORM,
        value=f'1xGX2570M5')
