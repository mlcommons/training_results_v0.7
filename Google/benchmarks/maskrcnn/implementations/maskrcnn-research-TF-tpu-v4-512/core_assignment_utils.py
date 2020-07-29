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
"""Core assignment utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from REDACTED.tensorflow.contrib import tpu as contrib_tpu


# The core assignments may change the performance of the model parallelism.
# An ideal case is to locate the op at the core that its input is located and
# avoid communication among cores.
# copybara:strip_begin
# This model has two stages: The first stage includes ResNet backbone, FPN, and
# RPN. The second stage includes a selection process (top-k, NMS, and ground
# truth assignment), Faster-RCNN, and Mask-RCNN. We apply spatial partition in
# the first stage and op parallelism in the second stage. Core assignment is the
# true source of op placement w.r.t. devices. Currently, this is hand-tuned and
# subject to change as the assumption (XLA scheduling/the placement of other
# ops) changes.
# copybara:strip_end
CORE_0 = 0
CORE_1 = 1
CORE_2 = 2


def get_core_assignment(core_assignment,
                        num_cores_per_replica=None,
                        use_spmd=False):
  """Returns core assignment based on the number of cores in a replica.

  When the model runs with model parallelism (i.e., multiple cores for one
  replica), the core assignment is a modular of the number of available cores.
  When the model runs with out model parallelism (`num_cores_per_replica` is
  None), the function returns `None` so that device placement is a no-op.

  Args:
    core_assignment: An `int` that represents the core number.
    num_cores_per_replica: An `int` that represents the number of cores. `None`
      means no model parallelism.
    use_spmd: A `bool` indicates if SPMD is enable.

  Returns:
    The core assignment based on whether the model runs with model parallelism
    and the number of cores per replica.
  """
  if num_cores_per_replica is not None and not use_spmd:
    return contrib_tpu.core(core_assignment % num_cores_per_replica)
  else:
    return None
