# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for the reinforcement trainer."""
import asyncio
import logging
import os
import sys

from absl import flags
import tensorflow as tf

from REDACTED.minigo import mask_flags
from REDACTED.minigo import utils as minigo_utils

sys.path.insert(0, '.')  # nopep8

# Flags that take multiple values.
# For these flags, expand_cmd_str appends each value in order.
# For all other flags, expand_cmd_str takes the last value.
MULTI_VALUE_FLAGS = set(['--lr_boundaries', '--lr_rates'])


flag_cache = {}
flag_cache_lock = asyncio.Lock()


def is_python_cmd(cmd):
  return cmd[0].startswith('python')


def get_cmd_name(cmd_list):
  path = cmd_list[1] if is_python_cmd(cmd_list) else cmd_list[0]
  return os.path.splitext(os.path.basename(path))[0]


async def expand_cmd_str(cmd):
  """Expands a string to run as command line argument."""
  real_args_position = 2 if is_python_cmd(cmd) else 1
  cmd = list(cmd)
  args = cmd[real_args_position:]
  process = cmd[:real_args_position]
  key = ' '.join(process)

  async with flag_cache_lock:
    valid_flags = flag_cache.get(key)
    if valid_flags is None:
      valid_flags = mask_flags.extract_valid_flags(cmd)
      flag_cache[key] = valid_flags

  parsed_args = flags.FlagValues().read_flags_from_files(args)
  flag_args = {}
  position_args = []
  for arg in parsed_args:
    if arg.startswith('--'):
      if '=' not in arg:
        flag_args[arg] = None
      else:
        flag, value = arg.split('=', 1)
        if flag in MULTI_VALUE_FLAGS:
          if flag not in flag_args:
            flag_args[flag] = []
          flag_args[flag].append(value)
        else:
          flag_args[flag] = value
    else:
      position_args.append(arg)

  flag_list = []
  for flag, value in flag_args.items():
    if value is None:
      flag_list.append(flag)
    elif isinstance(value, list):
      for v in value:
        flag_list.append('%s=%s' % (flag, v))
    else:
      flag_list.append('%s=%s' % (flag, value))

  flag_list = sorted(mask_flags.filter_flags(flag_list, valid_flags))
  return '  '.join(process + flag_list + position_args)


def list_selfplay_dirs(base_dir):
  """Returns a sorted list of selfplay data directories.

  Training examples are written out to the following directory hierarchy:
    base_dir/model_name/device_id/timestamp/

  Args:
    base_dir: either selfplay_dir or holdout_dir.

  Returns:
    A list of model directories sorted so the most recent directory is first.
  """

  model_dirs = [
      os.path.join(base_dir, x) for x in tf.io.gfile.listdir(base_dir)
  ]
  return sorted(model_dirs, reverse=True)


def copy_tree(src, dst, verbose=False):
  """Copies everything under src to dst."""

  print('Copying {} to {}'.format(src, dst))
  for src_dir, sub_dirs, basenames in tf.io.gfile.walk(src):
    rel_dir = os.path.relpath(src_dir, src)
    dst_dir = os.path.join(dst, rel_dir)
    for sub_dir in sorted(sub_dirs):
      path = os.path.join(dst, rel_dir, sub_dir)
      print('Make dir {}'.format(path))
      tf.io.gfile.makedirs(path)
    if basenames:
      print('Copying {} files from {} to {}'.format(
          len(basenames), src_dir, dst_dir))
      for basename in basenames:
        src_path = os.path.join(src_dir, basename)
        dst_path = os.path.join(dst_dir, basename)
        if verbose:
          print('Copying {} to {}'.format(src_path, dst_path))
        tf.io.gfile.copy(src_path, dst_path)


async def checked_run(cmd, env=None):
  """Run the given subprocess command in a coroutine.

  Args:
    cmd: the command to run and its arguments.
    env: optional OS environment to run the process under.

  Returns:
    The output that the command wrote to stdout & stderr.

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  # Start the subprocess.
  logging.info('Running: %s', await expand_cmd_str(cmd))
  with minigo_utils.logged_timer('{} finished'.format(get_cmd_name(cmd))):
    p = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT)

    # Stream output from the process stdout.
    lines = []
    while True:
      line = await p.stdout.readline()
      if not line:
        break
      line = line.decode()[:-1]
      lines.append(line)
      logging.info(line)

    # Wait for the process to finish, check it was successful & build stdout.
    await p.wait()
    output = '\n'.join(lines)[:-1]
    if p.returncode:
      raise RuntimeError('Return code {} from process: {}\n{}'.format(
          p.returncode, await expand_cmd_str(cmd), output))

    return output


def wait(awaitable_objs):
  """Waits for all of the awaitable objects (e.g.coroutines) in awaitable_objs to finish.

  All the awaitable objects are waited for, even if one of them raises an
  exception. When one or more awaitable raises an exception, the exception
  from the awaitable with the lowest index in the awaitable_objs list will be
  reraised.

  Args:
    awaitable_objs: a single awaitable, or list awaitables.

  Returns:
    If awaitable_objs is a single awaitable, its result.
    If awaitable_objs is a list of awaitables, a list containing the of each
    awaitable in the list.

  Raises:
      Exception: if any of the awaitables raises.
  """

  aws_list = awaitable_objs if isinstance(awaitable_objs,
                                          list) else [awaitable_objs]
  results = asyncio.get_event_loop().run_until_complete(
      asyncio.gather(*aws_list, return_exceptions=True))
  # If any of the cmds failed, re-raise the error.
  for result in results:
    if isinstance(result, Exception):
      raise result
  return results if isinstance(awaitable_objs, list) else results[0]
