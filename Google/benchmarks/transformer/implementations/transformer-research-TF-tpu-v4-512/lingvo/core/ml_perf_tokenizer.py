# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tokenizer for use for the MLPerf transformer benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unicodedata

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from REDACTED.transformer_lingvo.lingvo import compat as tf
from REDACTED.transformer_lingvo.lingvo.core import ops
from REDACTED.transformer_lingvo.lingvo.core import tokenizers

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
RESERVED_TOKENS = [PAD, EOS]

# Set of characters that will be used in the function _escape_token() (see func
# docstring for more details).
# This set is added to the alphabet list to ensure that all escaped tokens can
# be encoded.
_ESCAPE_CHARS = set(u"\\_u;0123456789")
# Regex for the function _unescape_token(), the inverse of _escape_token().
# This is used to find "\u", "\\", and "\###;" substrings in the token.
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")

_UNDEFINED_UNICODE = u"\u3013"

# Set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i)
    for i in xrange(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))

# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. The value is found using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1  # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 1000  # max value to use when binary searching for min_count


def _native_to_unicode(s):
  """Convert string to unicode (required in Python 2)."""
  if six.PY2:
    return s if isinstance(s, unicode) else s.decode("utf-8")
  else:
    return s


def _load_vocab_file(vocab_file, reserved_tokens=None):
  """Load vocabulary while ensuring reserved tokens are at the top."""
  reserved_tokens = []
  subtoken_list = []
  with tf.io.gfile.GFile(vocab_file, mode="r") as f:
    for line in f:
      subtoken = _native_to_unicode(line.strip())
      subtoken = subtoken[1:-1]  # Remove surrounding single-quotes
      if subtoken in reserved_tokens:
        continue
      subtoken_list.append(_native_to_unicode(subtoken))
  return reserved_tokens + subtoken_list


def _unicode_to_native(s):
  """Convert string from unicode to native format (required in Python 2)."""
  if six.PY2:
    return s.encode("utf-8") if isinstance(s, unicode) else s
  else:
    return s


def _unescape_token(token):
  r"""Replaces escaped characters in the token with their unescaped versions.

  Applies inverse transformations as _escape_token():
    1. Replace "\u" with "_", and "\\" with "\".
    2. Replace "\###;" with the unicode character the ### refers to.

  Args:
    token: escaped string

  Returns:
    unescaped string
  """

  def match(m):
    r"""Returns replacement string for matched object.

    Matched objects contain one of the strings that matches the regex pattern:
      r"\\u|\\\\|\\([0-9]+);"
    The strings can be '\u', '\\', or '\###;' (### is any digit number).

    m.group(0) refers to the entire matched string ('\u', '\\', or '\###;').
    m.group(1) refers to the first parenthesized subgroup ('###').

    m.group(0) exists for all match objects, while m.group(1) exists only for
    the string '\###;'.

    This function looks to see if m.group(1) exists. If it doesn't, then the
    matched string must be '\u' or '\\' . In this case, the corresponding
    replacement ('_' and '\') are returned. Note that in python, a single
    backslash is written as '\\', and double backslash as '\\\\'.

    If m.goup(1) exists, then use the integer in m.group(1) to return a
    unicode character.

    Args:
      m: match object

    Returns:
      String to replace matched object with.
    """
    # Check if the matched strings are '\u' or '\\'.
    if m.group(1) is None:
      return u"_" if m.group(0) == u"\\u" else u"\\"

    # If m.group(1) exists, try and return unicode character.
    try:
      return six.unichr(int(m.group(1)))
    except (ValueError, OverflowError) as _:
      return _UNDEFINED_UNICODE

  # Use match function to replace escaped substrings in the token.
  return _UNESCAPE_REGEX.sub(match, token)


def _join_tokens_to_string(tokens):
  """Join a list of string tokens into a single string."""
  token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
  ret = []
  for i, token in enumerate(tokens):
    if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
      ret.append(u" ")
    ret.append(token)
  return "".join(ret)


class MlPerfTokenizer(tokenizers.BaseTokenizer):
  """Id->String only for MLPerf decoding."""

  @classmethod
  def Params(cls):
    p = super(MlPerfTokenizer, cls).Params()
    p.Define("vocab_filepath", None, "Specifies a filepath to the vocab.")
    return p

  def IdsToStrings(self, ids, lens):
    p = self.params
    return ops.ml_perf_subword_id_to_string(
        ids, lens, vocab_filepath=p.vocab_filepath)

  def __init__(self, params):
    super(MlPerfTokenizer, self).__init__(params)
    reserved_tokens = RESERVED_TOKENS
    self.subtoken_list = _load_vocab_file(params.vocab_filepath,
                                          reserved_tokens)

    self.max_subtoken_length = 0
    for subtoken in self.subtoken_list:
      self.max_subtoken_length = max(self.max_subtoken_length, len(subtoken))

  def _subtoken_ids_to_tokens(self, subtokens):
    """Convert list of int subtoken ids to a list of string tokens."""
    escaped_tokens = "".join([
        self.subtoken_list[s] for s in subtokens if s < len(self.subtoken_list)
    ])
    escaped_tokens = escaped_tokens.split("_")

    # All tokens in the vocabulary list have been escaped (see _escape_token())
    # so each token must be unescaped when decoding.
    ret = []
    for token in escaped_tokens:
      if token:
        ret.append(_unescape_token(token))
    return ret

  def PythonIdsToStrings(self, ids, lens):
    """Unlike the normal IdsToStrings which is in-graph, this runs entirely in Python.

    Uses the reference MLPerf tokenizer code.

    Args:
      ids: A matrix of shape [batch, seqlen].
      ids[i, :] is the i-th sample's ids.
      lens: A vector of shape [batch]. lens[i] is the sequence length of the
          i-th sample. Only the first lens[i] tokens in ids[i, :] are valid
            tokens for the i-th sequence.

    Returns:
      A list of seqlen decoded strings.
    """
    resp = []
    for i, row in enumerate(ids):
      resp.append(self.decode(row[:lens[i]]))
    return resp

  def decode(self, subtokens):
    """Converts list of int subtokens ids into a string."""
    if isinstance(subtokens, np.ndarray):
      # Note that list(subtokens) converts subtokens to a python list, but the
      # items remain as np.int32. This converts both the array and its items.
      subtokens = subtokens.tolist()

    if not subtokens:
      return ""

    assert isinstance(subtokens, list) and isinstance(subtokens[0], int), (
        "Subtokens argument passed into decode() must be a list of integers.")

    return _unicode_to_native(
        _join_tokens_to_string(self._subtoken_ids_to_tokens(subtokens)))
