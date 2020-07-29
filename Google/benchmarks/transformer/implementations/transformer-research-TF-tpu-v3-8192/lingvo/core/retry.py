# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Retry on exception."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from REDACTED import retry


def Retry(retry_value=Exception,
          max_retries=None,
          initial_delay_sec=0.1,
          delay_growth_factor=1.1,
          delay_growth_fuzz=0.1,
          max_delay_sec=60,
          **kwargs):
  """Returns a retry decorator."""
  if max_retries is None:
    max_retries = 2**30  # Effectively forever.
  return retry.logged_retry_on_exception(
      retry_value=retry_value,
      retry_intervals=retry.FuzzedExponentialIntervals(
          num_retries=max_retries,
          initial_delay_sec=initial_delay_sec,
          factor=delay_growth_factor,
          fuzz=delay_growth_fuzz,
          max_delay_sec=max_delay_sec),
      **kwargs)
