import collections
import os
import subprocess
import torch

from mlperf_logging.mllog import constants
from seq2seq.utils import configure_logger, log_event


def mlperf_submission_log(benchmark):

    num_nodes = os.environ.get('SLURM_NNODES', 1)

    configure_logger(benchmark)

    log_event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    log_event(
        key=constants.SUBMISSION_ORG,
        value='Inspur')

    log_event(
        key=constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        key=constants.SUBMISSION_STATUS,
        value='onprem')

    log_event(
        key=constants.SUBMISSION_PLATFORM,
        value=f'{num_nodes}xNF5488')
