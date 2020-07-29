# Lint as: python2, python3
"""Base class for all jobs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import base_runner as lingvo_base_runner
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import compat as tf

from REDACTED.learning.REDACTED.research.babelfish import REDACTED_utils
#from third_party.tensorflow_models.mlperf.models.rough.transformer_lingvo.google import REDACTED_utils


class BaseRunner(lingvo_base_runner.BaseRunner):
  """Base class for all jobs."""

  def _SetStatusMessage(self, message, retrying=False):
    """Update the REDACTED status message for this task."""
    REDACTED_utils.SetBorgStatusMessage(
        self._FormatStatusMessage(message, retrying))

  def _CreateSummaryWriter(self, logdir):
    """Creates and returns a tf summary writer."""
    suffix = None
    return tf.summary.FileWriter(logdir, filename_suffix=suffix)
