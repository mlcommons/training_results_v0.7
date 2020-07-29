# Lint as: python2, python3
r"""Babelfish version of Beam search.

This is TensorFlow implementation of beam search that is implemented to run fast
on TPUs. go/beam-search-on-fish
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from REDACTED.transformer_lingvo.lingvo import compat as tf
from six.moves import range
from REDACTED.learning.REDACTED.google.xla.python import xla_ops

# A large negative constant to match babelfish cpu beam search decoders init
# value
BEST_SCORES_INIT = -1e36

_MAX_BFLOAT16_INT = 1 << 8
_MAX_INT32 = (1 << 31) - 1


# Checks whether we can represent val in bfloat16. It keeps 1 bit for the sign,
# 8 bits for the exponent and 7 bits for the mantissa. Largest integer we can
# represent exactly is 2^(mantissa+1).
def _has_bfloat16_repr(val):
  return val is not None and val <= _MAX_BFLOAT16_INT


class _Gatherer(object):
  """Helper for fast_gather() that delegates to various implementations."""

  def __init__(self, ids, ids_size):
    self._ids = ids
    self._ids_size = ids_size

  def __call__(self, *args, **kwargs):
    return self.gather(*args, **kwargs)

  def gather(self, values, max_value=None, axis=0, batch_major_state=True):
    """Returns 'values' gathered at the ids provided to the constructor.

    Args:
      values: Values to gather from.
      max_value: The largest of values.
      axis: Axis to gather on. Defaults to 0 (rows).
      batch_major_state: Whether the values to gather from use batch major or
        not. Defaults to True. For Transformer model, batch_major_state is set
        to False (time is the major dim).

    Returns:
      Gathered values.

    Raises:
      Value error: If dtype is not supported.
      NotImplemented error: if axis is not 0 or 1.
    """
    # Carry out the gather via matmul if the values are floating point or can
    # be represented exactly in bfloat16 (TPU internal matmul dtype).
    dtype = values.dtype
    if dtype in (tf.bfloat16, tf.float32,
                 tf.bool) or _has_bfloat16_repr(max_value):
      return self._matmul_gather(
          values, axis=axis, batch_major_state=batch_major_state)
    elif dtype == tf.int32:
      # For int32s with a max_value that can't be represented exactly in
      # floating point, we decompose `values` into parts that can be represented
      # exactly, gather each part individually, and recombine to get the final
      # gathered values.
      max_value = max_value or _MAX_INT32
      if max_value <= _MAX_BFLOAT16_INT**2:
        # Break 'values' into two bfloat16-representable parts. The low part
        # values are in [-255, 255]. High part values are in [-256, 256].
        signs = tf.sign(values)
        abs_values = tf.abs(values)
        low_part = signs * tf.bitwise.bitwise_and(abs_values, 0xff)
        high_part = signs * tf.bitwise.right_shift(abs_values, 8)
        low_part_gathered = self._matmul_gather(
            low_part, axis=axis, batch_major_state=batch_major_state)
        high_part_gathered = self._matmul_gather(
            high_part, axis=axis, batch_major_state=batch_major_state)
        return tf.bitwise.left_shift(high_part_gathered, 8) + low_part_gathered
      else:
        # For larger magnitude int32s, we could break them up into 3 or 4 byte-
        # sized matmuls, but regular-old tf.gather() is more efficient.
        return tf.gather(values, self._ids, axis=axis)
    else:
      raise ValueError("Unsupported dtype %s" % values.dtype)

  def _matmul_gather(self, values, axis=0, batch_major_state=True):
    """Returns values gathered.

    Args:
      values: Values to gather from.
      axis: Axis to gather on. Defaults to 0 (rows).
      batch_major_state: Whether the values to gather from use batch major or
        not. Defaults to True. For Transformer model, batch_major_state is set
        to False (time is the major dim).

    Returns:
      Gathered values.

    Raises:
      NotImplemented error if axis is not 0 nor 1.
    """

    dtype = values.dtype
    if dtype != tf.float32 and dtype != tf.bfloat16:
      values = tf.cast(values, tf.float32)

    if axis == 0:
      if values.shape.rank is not None and values.shape.rank > 2:
        if not batch_major_state:
          values = tf.transpose(values, [1, 0, 2])
        results = tf.cast(
            tf.gather(values, tf.cast(self._ids, tf.int32)), dtype)
        # pylint:disable=g-long-ternary
        return (tf.transpose(results, [1, 0, 2])
                if not batch_major_state else results)
        # pylint:enable=g-long-ternary
      else:
        one_hot_ids = tf.one_hot(self._ids, self._ids_size, dtype=values.dtype)
        return tf.cast(tf.matmul(one_hot_ids, values), dtype)
    elif axis == 1:
      one_hot_ids = tf.one_hot(
          self._ids, self._ids_size, dtype=values.dtype, axis=0)
      return tf.cast(tf.matmul(values, one_hot_ids), dtype)
    else:
      raise NotImplementedError("Only row/col-wise gather implemented.")


def fast_gather(values,
                ids,
                ids_size,
                max_value=None,
                axis=0,
                batch_major_state=True):
  """Fast implementation of gather on TPUs.

  Args:
    values: Values to gather from.
    ids: ids (rows to gather)
    ids_size: id space size.
    max_value: Optional hint on maximum value for int32 that allows to speed up
      the gather operation.
    axis: axis to gather on. Defaults to 0 (rows).
    batch_major_state: Whether the values to gather from use batch major or not.
      Defaults to True.

  Returns:
    Gathered values.
  Raises:
    Value error if values is type int64.
  """
  values = tf.convert_to_tensor(values)
  ids = tf.convert_to_tensor(ids)
  with tf.name_scope("fast_gather"):
    return _Gatherer(ids, ids_size)(
        values,
        max_value=max_value,
        axis=axis,
        batch_major_state=batch_major_state)


def reorder_tensor(reorder_mode,
                   values,
                   num_shards,
                   shard_size,
                   max_value=None,
                   axis=0):
  """Reorder tensor based on the mode passed in.

  This method reorders rows or cols (based on `axis`) of the tensor passed in
  from one sharding mode to another sharding mode. This method uses matmul for
  reordering to be efficient on TPUs.

  Args:
    reorder_mode: Either mod_to_div or div_to_mod
    values: Tensor to reorder
    num_shards: Number of shards.
    shard_size: Size of each shard.
    max_value: If dtype=tf.int32, and we know maximum of the values, we can
      efficiently implement it as matmuls.
    axis: axis to gather on. Defaults to 0 (rows).

  Returns:
    A tensor of same shape as values but rows (or first axis) reordered.
  """
  values = tf.convert_to_tensor(values)
  with tf.name_scope("reorder_tensor_" + reorder_mode):
    num_ids = num_shards * shard_size
    # Elements to gather.
    seq_ids = tf.range(num_ids)
    if reorder_mode == "mod_to_div":
      local_ids = seq_ids // shard_size
      shard_ids = seq_ids % shard_size
      ids = local_ids + shard_ids * num_shards
    elif reorder_mode == "div_to_mod":
      shard_ids = seq_ids % num_shards
      local_ids = seq_ids // num_shards
      ids = local_ids + shard_ids * shard_size
    else:
      raise NotImplementedError(
          "Reorder mode: {} not implemented.".format(reorder_mode))
    return fast_gather(values, ids, num_ids, max_value, axis=axis)


def _log_sum_exp(a, b):
  m = tf.maximum(a, b)
  return m + tf.math.log(tf.exp(a - m) + tf.exp(b - m))


def merge_hyps(global_score_values, histories_in, mask, num_beams,
               num_hyps_per_beam):
  """Merges candidate hypotheses with identical histories.

  This function takes a set of candidate hypotheses, represented as Tensors of
  scores and histories, and merges all pairs of hypotheses that have identical
  history hashes. When two hypotheses are merged, the hyp with lower global
  score gets "deleted" and has its probability mass added to the higher scoring
  one. Hypotheses are "deleted" by giving them empty history and a large
  negative global score. The function output is a tuple of new
  (global_score_values, histories) Tensors.

  All input Tensors are assumed to be in "div" hypothesis ordering. That is,
  element [i, ...] corresponds to the j-th hyp of the n-th beam, where j = i % k
  and n = i / k.

  Example:
    Suppose num_beams = 1, num_hyps_per_beam = 2, candidates_per_hyp = 5,
    global_score_values is
      [[11 12 13 14 15],
       [17 16 10 19 20]]
    and histories_in is
      [[1 2 3 4 5],
       [5 6 3 7 8]].

    There are two pairs of hypotheses with identical histories that should
    be merged -- two with hash value 3 and two with hash 5. In each pair, the
    one with lower score will be deleted and merged into the one with higher
    score.

    The output is a new set of global_score_values,
      [[ 11     12 13.04 14 -1e34 ],
         17.13  16 -1e34 19 20    ]]
    and new histories
      [[1 2 3 4 0],
       [5 6 0 7 8]].
    Hypotheses deleted in the merge now have zero history and a large negative
    score. The destination of each merge now has additional probability mass.
    (Note _log_sum_exp(13, 10) ~= 13.04 and _log_sum_exp(15, 17) ~= 17.13.)

  Args:
    global_score_values: Tensor of shape [b * k, candidates_per_hyp], the global
      scores of each candidate hypothesis.
    histories_in: int32 Tensor of shape [b * k, candidates_per_hyp], the
      histories of each candidate hypothesis.
    mask: Tensor of shape [b * k, 1] indicating which entries in
      global_score_values and histories_in are valid.
    num_beams: int, the number of beams (b above).
    num_hyps_per_beam: int, the number of hypotheses per beam (k above).

  Returns:
    A tuple of new (global_score_values, histories) updated so that input
    hypotheses with identical histories are now merged. Hypotheses deleted in
    the merge have a new global score of BEST_SCORES_INIT and a history of 0.
  """
  values_dtype = global_score_values.dtype
  candidates_per_hyp = histories_in.get_shape()[1]
  k = num_hyps_per_beam

  # High-level strategy: To detect hyps to merge, we'll permute the hypotheses
  # within each beam so that their histories are in sorted order. We can then
  # in parallel check whether each history is equal to its left or right
  # neighbor (i.e. whether the hyps should be merged), and if so, which of them
  # has the higher global score (the direction of the merge). When two hyps need
  # to be merged, we'll "delete" the one with lower score (by giving it a large
  # negative score and empty history) and add its probability mass to the other.
  #
  # Note we only have to do pair-wise merging once per beam search step, because
  # (ignoring hash collisions) there are at most two candidate hypotheses with
  # any particular history. This follows from the fact that hypotheses are
  # unique at the start of the beam search step, as are the top K non-epsilon
  # extensions of those hypotheses. Thus, if there are two paths with
  # identical histories, they must have the form
  #   h_i <eps> == h_j s  (for some i != j, s != eps),
  # where h_i and h_j are distinct input hypotheses, and s is some non-epsilon
  # symbol.

  # Reshape inputs to [b, num_hyps_per_beam * candidates_per_hyp] so they're
  # grouped by beam.
  histories = histories_in
  orig_scores_shape = tf.shape(global_score_values)
  histories = tf.reshape(histories, [num_beams, -1])
  histories_valid = tf.cast(
      tf.reshape(tf.tile(mask, [1, candidates_per_hyp]), [num_beams, -1]),
      values_dtype)
  # Compute the permutation of hyps within each beam that put the histories in
  # sorted order, and the one that permutates the sorted hyps back to the
  # original order.
  sorted_history_indices = tf.argsort(histories, axis=1)
  inverse_indices = tf.argsort(sorted_history_indices, axis=1)

  def to_flat_indices(column_indices_per_row):
    column_indices_per_row.shape.assert_has_rank(2)
    flat_indices = (
        column_indices_per_row + num_hyps_per_beam * candidates_per_hyp *
        tf.reshape(tf.range(num_beams), [num_beams, 1]))
    return tf.reshape(flat_indices, [-1])

  # Convert to linear indices so we can use fast_gather.
  sorted_history_indices_flat = to_flat_indices(sorted_history_indices)
  inverse_indices_flat = to_flat_indices(inverse_indices)

  def history_sort(values):
    return tf.reshape(
        fast_gather(
            tf.reshape(values, [-1, 1]), sorted_history_indices_flat,
            num_beams * k * candidates_per_hyp),
        [num_beams, k * candidates_per_hyp])

  def history_unsort(values):
    return tf.reshape(
        fast_gather(
            tf.reshape(values, [-1, 1]), inverse_indices_flat,
            num_beams * k * candidates_per_hyp), orig_scores_shape)

  sorted_histories = history_sort(histories)
  sorted_histories_valid = history_sort(histories_valid)

  # Indicators of whether each hypothesis is a duplicate of its left/right
  # neighbors.
  # [num_batches, k * candidates_per_hyp - 1]
  dup_mask = tf.cast(
      tf.equal(sorted_histories[:, 1:], sorted_histories[:, :-1]),
      values_dtype) * (
          sorted_histories_valid[:, 1:] * sorted_histories_valid[:, :-1])
  padding = tf.zeros([num_beams, 1], dtype=values_dtype)
  is_dup_of_left = tf.concat([padding, dup_mask], axis=1)
  is_dup_of_right = tf.concat([dup_mask, padding], axis=1)

  # Examine global scores to see which hyps should be merged, and within those
  # cases, which hyps get deleted/retained in the merge.
  sorted_global_scores = history_sort(global_score_values)
  # Global scores of each hyp's left and right neighbors.
  right_global_scores = tf.concat([sorted_global_scores[:, 1:], padding],
                                  axis=1)
  left_global_scores = tf.concat([padding, sorted_global_scores[:, :-1]],
                                 axis=1)

  # Masks indicating whether each candidate hyp is better or worse than its
  # left or right neighbor.
  is_better_than_right = tf.cast(
      tf.greater_equal(sorted_global_scores, right_global_scores), values_dtype)
  is_worse_than_right = 1.0 - is_better_than_right
  is_better_than_left = tf.cast(
      tf.greater(sorted_global_scores, left_global_scores), values_dtype)
  is_worse_than_left = 1.0 - is_better_than_left

  # Determine which hypotheses need to be merged.
  is_merge_source = tf.minimum(
      is_dup_of_left * is_worse_than_left +
      is_dup_of_right * is_worse_than_right, 1.0)
  is_left_merge_dest = is_dup_of_left * is_better_than_left
  is_right_merge_dest = is_dup_of_right * is_better_than_right
  is_merge_dest = tf.minimum(is_left_merge_dest + is_right_merge_dest, 1.0)
  # Mask of hyps unaffected by merging.
  is_unchanged = tf.maximum(1.0 - is_merge_source - is_merge_dest, 0.0)

  sorted_global_scores = (
      is_unchanged * sorted_global_scores + is_merge_source * BEST_SCORES_INIT +
      is_left_merge_dest *
      _log_sum_exp(left_global_scores, sorted_global_scores) +
      is_right_merge_dest *
      _log_sum_exp(right_global_scores, sorted_global_scores))
  # Set histories of deleted (merge source) hyps to zero.
  sorted_histories *= tf.cast(1.0 - is_merge_source, sorted_histories.dtype)

  # Put everything back in its original order and rank.
  global_score_values_out = history_unsort(sorted_global_scores)
  histories_out = history_unsort(sorted_histories)
  return global_score_values_out, histories_out


# pylint: disable=g-doc-return-or-yield
def _hash32(x, seed=1):
  """A simple int32 -> int32 hash function.

  Args:
    x: int32 Tensor, the value to hash.
    seed: int or int32 Tensor, the seed(s) values. Must be broadcastable to the
      shape of 'x'.
  """
  x = tf.convert_to_tensor(x)
  assert x.dtype == tf.int32
  # TODO(austinwaters): Change to an int64 valued hash once fast_gather
  # supports that type. For int64, the corresponding CityHash prime (k_mul) is
  # 0x9ddfea08eb382d69, and we should right shift by 32 bits in the last line.
  k_mul = 0xcc9e2d51  # Large prime borrowed from CityHash.
  m = (seed + x) * k_mul
  return tf.bitwise.bitwise_xor(m, (tf.bitwise.right_shift(m, 16)))
# pylint: enable=g-doc-return-or-yield


def _hash32_combine(a, b):
  return _hash32(a, seed=b)


def update_histories(histories_in,
                     new_symbol_ids,
                     histories_valid_mask,
                     epsilon_id=0):
  """Updates hypothesis histories to account for appending a new symbol.

  Context: Path merging requires a mechanism to compare whether any two
  hypothesized sequences are equivalent after the removal of any epsilon
  symbols. To do such comparisons efficiently on TPU, we represent each
  hypothesis sequence by an int32 hash, referred to in this code simply as a
  "history". Histories are updated in a step-by-step manner as beam search
  progresses, in a manner that is invariant to the presence of epsilons. For
  example, the history hashes of "a <eps> b" and "a b <eps>" are equal.

  This function takes the histories for the current set of hypotheses
  (histories_in) and returns new histories that reflect the addition of one more
  symbol (new_symbol_ids).

  On the initial step of beam search, histories_in is assumed to contain all
  zeros. This is because this code uses the zero history to represent an
  empty hypothesis, i.e. one with no non-epsilon symbols.

  Args:
    histories_in: int32 Tensor of shape [b * k, 1] containing the histories of
      current hypotheses. Should be all zero if called during the initial step
      of beam search.
    new_symbol_ids: int32 Tensor of shape [b * k, num_candidates]. Slice
      [i, :] contains num_candidates symbol ids with which to extend the hyp
        represented by histories_in[i].
    histories_valid_mask: Tensor of shape [b * k, 1] indicating which entries of
      histories_in are valid.
    epsilon_id: int, the id for the epsilon symbol.

  Returns:
    New histories as an int32 Tensor of shape [b * k, num_candidates]. Element
    [i, j] corresponds to history_in[i, 1] extended by symbol
    new_symbol_ids[i, j].
  """
  histories_in.shape.assert_has_rank(2)
  new_symbol_ids.shape.assert_has_rank(2)
  histories_valid_mask.shape.assert_has_rank(2)
  assert epsilon_id >= 0

  hist_dtype = histories_in.dtype
  with tf.name_scope("update_histories"):
    epsilon_id = tf.cast(epsilon_id, hist_dtype)
    new_sym_is_epsilon = tf.cast(
        tf.equal(new_symbol_ids, epsilon_id), hist_dtype)

    # The hash of each new symbol, or 0 (empty history) if epsilon.
    new_symbol_hashes = _hash32(new_symbol_ids) * (1 - new_sym_is_epsilon)
    history_is_empty = tf.cast(tf.equal(histories_in, 0), hist_dtype)

    # Compute new histories, respecting empty histories and epsilons in the
    # new_symbol_ids, as follows:
    #                   history_in is empty   | is not empty
    # new_symbol_id     -----------------------------------------------
    #   is eps            <empty history>     | history_in
    #   is not eps       hash(symbol_id)      | hash_combine(history_in,
    #                                         |              hash(symbol_id))
    histories = (1 - history_is_empty) * (
        histories_in * new_sym_is_epsilon +
        _hash32_combine(histories_in, new_symbol_hashes) *
        (1 - new_sym_is_epsilon)) + history_is_empty * new_symbol_hashes

    # Set invalid entries to zero, the empty history.
    histories *= tf.cast(histories_valid_mask, hist_dtype)
    return histories


def beam_search_step(in_scores,
                     in_atten_probs,
                     in_best_scores,
                     in_cumulative_scores,
                     in_histories,
                     cur_step,
                     eos_id,
                     num_beams,
                     beam_size,
                     num_hyps_per_beam,
                     valid_eos_max_logit_delta=5.0,
                     local_eos_threshold=-100.0,
                     merge_paths=False,
                     is_last_chunk=None,
                     eoc_id=0):
  """A single step of beam search.

  Let "b" be the number of beams, "k" be the number hyps in each beam. This
  function supports values with dtypes tf.float32 or tf.bfloat16.

  The following data structures are allocated before the first decoding step and
  are passed along from cur step to the next step:

  Args:
    in_scores: A tensor of shape [b * k, vocab_size], where [i, ...] is the
      token score of the j-th hyps of the n-th beam. j = (i / k), and n = i % k
    in_atten_probs: A tensor of shape [b*k, s_len], where in_atten_probs[i, ...]
      is the attention probabilities over the source words of the j-th hyps of
      n-th beam (where j, and n are derived as above).
    in_best_scores: A vector of size [b], best scores of terminated hyps so far
      in each of the beams.
    in_cumulative_scores: A vector of size [b * k]. The cumulative score of each
      active hyp before the current step.
    in_histories: An int32 vector of size [b * k] containing hashes of the
      histories of each active hyp. If 'merge_paths' is enabled, the histories
      are used to identify hypotheses that are identical modulo epsilons (e.g.
      "a <eps> b" and "a b <eps>") and merge them. See 'update_histories'
      docstring for details.
    cur_step: Current step id.
    eos_id: Token id of the special end of sequence token.
    num_beams: Number of beams.
    beam_size: Search terminates if the delta between the scores of the active
      hyps.
    num_hyps_per_beam: Number of hyps in a beam.
    valid_eos_max_logit_delta: We allow </s> to terminate a hyp only if its
      logit is no more than 'valid_eos_max_logit_delta' away from the logit of
      the best candidate.
    local_eos_threshold: We allow </s> to terminate a hyp if the local score for
      </s> is greater than local_eos_threshold.
    merge_paths: If true, hyps which are identical when epsilons are removed
      will be combined into a single hyp.  The probability for that combined hyp
      will be the sum of the probabilities of the component hyps.  This can only
      be applied for epsilon-emitting models (RNN-T and NT).
    is_last_chunk: A tensor of shape [b * k, 1]. Used by neural transducer,
      determines whether the current hypothesis reaches the last chunk and
      should treat the next end-of-chunk symbol as end-of-sentence.
    eoc_id: int, the id of the end of chunk (a.k.a epsilon) token used by neural
      transducer models. Only relevant if 'merge_paths' is True or
      'is_last_chunk' is provided.

  Returns:
    out_best_scores: A tensor of shape [b] of updated best scores for each of
      the beams.
    out_cumulative_scores: A tensor of shape [b * k]. The cumulative score of
      the new hyps after the current decoding step.
    out_scores: A tensor of shape [b * k] with scores of the token selected.
    out_eos_scores: A tensor of shape [b * k] with token scores for the EOS, in
      case the hyp was terminated, otherwise 0.0.
    out_hyps: A tensor of shape [b * k] with ids of the token selected.
    out_prev_hyps: A tensor of shape [b * k] with index to the previous hyps
      which was selected.
    out_done_hyps: A boolean tensor of shape [b * k] where value indicates
      if hyps was terminated.
    out_atten_probs: A tensor of shape [b * k, seq_len] which contain the
      attention probabilities over the source words against word in the previous
      hyps.
    out_eos_atten_probs: A tensor of shape [b * k, seq_len] which contains the
      attention probabilities over the source against word in the current hyp
      which was terminated.
    out_all_done: A scalar, whether decoding should terminate for all beams.
    out_histories: A tensor of shape [b * k] containing new history hashes for
      the active hypotheses. See 'update_histories' docstring for details.
  Raises:
    ValueError: if inputs are invalid.
  """
  num_hyps_per_beam = int(num_hyps_per_beam)

  if num_hyps_per_beam <= 0:
    raise ValueError(
        "num_hyps_per_beam = {} and must be > 0.".format(num_hyps_per_beam))

  in_scores = tf.convert_to_tensor(in_scores)
  in_scores.shape.assert_has_rank(2)
  num_classes = in_scores.get_shape()[1]

  in_atten_probs = tf.convert_to_tensor(in_atten_probs)
  in_atten_probs.shape.assert_has_rank(2)

  in_best_scores = tf.convert_to_tensor(in_best_scores)
  in_best_scores.shape.assert_has_rank(1)

  in_cumulative_scores = tf.convert_to_tensor(in_cumulative_scores)
  in_cumulative_scores.shape.assert_has_rank(1)

  in_histories = tf.convert_to_tensor(in_histories)
  in_histories.shape.assert_has_rank(1)

  with tf.name_scope("beam_search_step"):
    # For k = num_hyps_per_beam
    # First step of beam search is to find the top tokens based on its score.
    # Normally we select k+1, where the extra +1 is to make sure we have k
    # non-eos tokens to select if EOS token is in the top-k. If path merging is
    # on, we actually need to select k+2; this ensures there are k+1 tokens left
    # after the merge, at least k of which are not EOS.
    # TODO(b/118644069): Avoid casts when there is a XLA op available that takes
    # in bfloat16.
    num_candidates_per_input_hyp = (
        num_hyps_per_beam + 2 if merge_paths else num_hyps_per_beam + 1)
    # [b * k, num_candidates_per_input_hyp]
    local_score_values, local_indices = xla_ops.top_k_with_unique(
        tf.cast(in_scores, tf.float32), k=num_candidates_per_input_hyp)
    local_score_values = tf.cast(local_score_values, in_scores.dtype)

    # Compute the global score which is sum of the local score, and the
    # cumulative scores for each of the hyps.
    # [b * k, num_candidates_per_input_hyp]
    global_score_values = local_score_values + tf.expand_dims(
        in_cumulative_scores, 1)

    values_dtype = local_score_values.dtype
    is_first_step = tf.cast(tf.equal(cur_step, 0), values_dtype)

    # Preprocessing to reorder the tensor from `mod` sharding to `div` so that
    # we can use matrix/vector operations to complete the beam search.
    # [b * k, num_candidates_per_input_hyp]
    global_score_values = reorder_tensor("mod_to_div", global_score_values,
                                         num_beams, num_hyps_per_beam)
    local_score_values = reorder_tensor("mod_to_div", local_score_values,
                                        num_beams, num_hyps_per_beam)
    local_indices = reorder_tensor(
        "mod_to_div",
        local_indices,
        num_beams,
        num_hyps_per_beam,
        max_value=num_classes - 1)
    # [b * k, 1]
    histories = reorder_tensor("mod_to_div", tf.expand_dims(in_histories, 1),
                               num_beams, num_hyps_per_beam)
    if is_last_chunk is None:
      is_last_chunk = tf.zeros([num_beams * num_hyps_per_beam, 1], tf.bool)
    else:
      is_last_chunk = tf.cast(
          reorder_tensor(
              "mod_to_div",
              tf.reshape(is_last_chunk, [num_beams * num_hyps_per_beam, 1]),
              num_beams, num_hyps_per_beam), tf.bool)

    # For the first step mask everything but the first row.
    # [num_hyps_per_beam]
    per_example_mask = tf.concat([
        tf.constant([1.0], dtype=values_dtype),
        tf.zeros([num_hyps_per_beam - 1], dtype=values_dtype)
    ], 0)
    # [num_hyps_per_beam, num_beams] => [b*k, 1]
    mask = tf.reshape(
        tf.tile(per_example_mask, tf.expand_dims(num_beams, 0)),
        [-1, 1]) * is_first_step + (1.0 - is_first_step)
    local_score_values *= mask
    global_score_values *= mask

    # We add a large negative value for the unmasked values.
    per_example_additive_mask = tf.concat([
        tf.constant([0.0], dtype=values_dtype),
        tf.constant(
            BEST_SCORES_INIT, shape=[num_hyps_per_beam - 1], dtype=values_dtype)
    ], 0)
    additive_mask = tf.reshape(
        tf.tile(per_example_additive_mask, tf.expand_dims(num_beams, 0)),
        [-1, 1]) * is_first_step
    local_score_values += additive_mask
    global_score_values += additive_mask

    if merge_paths:
      with tf.name_scope("merge_paths"):
        # Compute new history hashes for each hypothesis + new token.
        # [b * k, num_candidates_per_input_hyp]
        histories = update_histories(
            histories, local_indices, mask, epsilon_id=eoc_id)
        global_score_values, histories = merge_hyps(global_score_values,
                                                    histories, mask, num_beams,
                                                    num_hyps_per_beam)

    # As we keep num_candidates_per_input_hyp, we have a total of
    # num_candidates_per_input_hyp * k hyps active per example.
    num_candidate_hyps = num_candidates_per_input_hyp * num_hyps_per_beam
    batch_shape = [-1, num_candidate_hyps]

    # Reshape score values so that each row corresponds to a particular example.
    # [num_beams, num_candidate_hyps]
    global_score_values_batch = tf.reshape(global_score_values, batch_shape)

    # First for each beam: Find the top 2 * num_hyps_per_beam candidates.
    # The factor of 2 is to be able to process non EOS token ids in the case
    # where top scoring token for each hyps is EOS token.
    # [k * b, 2 * num_hyps_per_beam]
    _, candidates_indices_in_top_k = xla_ops.top_k_with_unique(
        tf.cast(global_score_values_batch, tf.float32), k=2 * num_hyps_per_beam)
    # Find the previous hyps of the candidate. We divide here by (k+1) to
    # identify which hyps this token came from.
    hyps_id = candidates_indices_in_top_k // num_candidates_per_input_hyp

    # Add in offset so that we can get the candidate index in the [b * k] space.
    offset = tf.expand_dims(tf.range(num_beams) * num_candidate_hyps, 1)
    flat_candidates_indices_in_top_k = tf.reshape(
        candidates_indices_in_top_k + offset, [-1])

    flat_local_indices = tf.reshape(local_indices, [1, -1])
    flat_token_scores = tf.reshape(local_score_values, [-1, 1])
    flat_global_scores = tf.reshape(global_score_values, [-1, 1])

    # Gather the token scores for each of 2*k candidates. We use tf.one_hot()
    # followed by a tf.matmul() to speedup gather on TPUs.
    total_num_candidates = num_beams * num_candidate_hyps
    token_scores_for_beam = tf.reshape(
        fast_gather(flat_token_scores, flat_candidates_indices_in_top_k,
                    total_num_candidates), [num_beams, 2 * num_hyps_per_beam])
    token_scores_for_beam_shape = tf.shape(token_scores_for_beam)

    global_scores_for_beam = tf.reshape(
        fast_gather(flat_global_scores, flat_candidates_indices_in_top_k,
                    total_num_candidates), token_scores_for_beam_shape)

    # Local indices value's are between [0, vocab_size-1], hence we use the
    # slower version of gather.
    token_ids_for_beam = tf.reshape(
        fast_gather(
            flat_local_indices,
            flat_candidates_indices_in_top_k,
            total_num_candidates,
            max_value=num_classes - 1,
            axis=1), token_scores_for_beam_shape)

    # We have access to 2*num_hyps_per_beam hyps per beam.
    # We shrink back to num_hyps_per_beam that does not include EOS, and move
    # EOS that occurs in top-num_hyps_per_beam to the EOS done matrix.

    # To determine the threshold at which eos is allowed to terminate a hyp,
    # we need to know the maximum global score for that hyp with any additional
    # token. If path merging is *not* enabled, the global_score_values are
    # by construction in sorted order, so we can just look at its 0th column. If
    # path merging is enabled, the global scores of deleted (merged) hyps break
    # the sorted order, which means we have to do a full reduce_max.
    if merge_paths:
      max_global_score_per_input_hyp = tf.reduce_max(
          global_score_values, axis=1, keepdims=True)
    else:
      max_global_score_per_input_hyp = global_score_values[:, 0:1]
    # [num_beams * num_hyps_per_beam, 1]
    global_eos_threshold = (
        max_global_score_per_input_hyp - valid_eos_max_logit_delta)
    local_eos_threshold_tensor = local_eos_threshold * tf.ones_like(
        global_eos_threshold)

    # Find EOS in top num_hyps_per_beam token ids. We also treat EOC as EOS if
    # the model has indicated this is the last chunk.
    local_index_is_eos = tf.equal(local_indices, eos_id)
    local_index_is_last_chunk_eoc = tf.math.logical_and(
        tf.equal(local_indices, eoc_id), is_last_chunk)
    eos_mask = tf.math.logical_and(
        tf.math.logical_and(
            tf.math.logical_and(
                tf.greater(
                    local_score_values,
                    tf.tile(local_eos_threshold_tensor,
                            [1, num_candidates_per_input_hyp])),
                tf.greater(
                    global_score_values,
                    tf.tile(global_eos_threshold,
                            [1, num_candidates_per_input_hyp]))),
            tf.math.logical_or(local_index_is_eos,
                               local_index_is_last_chunk_eoc)),
        tf.cast(mask, tf.bool))
    end_hyps_bool_mask = tf.reshape(tf.reduce_any(eos_mask, 1), [-1, 1])

    end_hyps_bool_mask = reorder_tensor("div_to_mod", end_hyps_bool_mask,
                                        num_beams, num_hyps_per_beam)

    eos_atten_probs = in_atten_probs * tf.cast(end_hyps_bool_mask,
                                               in_atten_probs.dtype)
    eos_atten_probs = tf.reshape(eos_atten_probs,
                                 [num_beams * num_hyps_per_beam, -1])
    # A boolean tensor of shape [b * k] where value indicates if hyps was
    # terminated.
    out_done_hyps = tf.reshape(end_hyps_bool_mask, [-1])

    # Scores for EOS token.
    eos_float_mask = tf.cast(eos_mask, values_dtype)
    eos_local_scores = eos_float_mask * local_score_values
    eos_additive_float_mask = (1.0 - eos_float_mask) * BEST_SCORES_INIT
    eos_local_scores += eos_additive_float_mask
    out_eos_scores = tf.reshape(tf.reduce_max(eos_local_scores, 1), [-1, 1])
    out_eos_scores = tf.reshape(
        reorder_tensor("div_to_mod", out_eos_scores, num_beams,
                       num_hyps_per_beam), [-1])
    # A tensor of shape [b] of updated best scores for each of the beams.
    eos_global_scores = eos_float_mask * global_score_values
    eos_global_scores += eos_additive_float_mask
    best_scores = tf.reduce_max(
        tf.reshape(eos_global_scores, [num_beams, -1]), 1)

    # Following operations are to finds the top num_hyps_per_beam that are
    # active.

    # Active ones are the ones that do not correspond to EOS termination.
    # We keep num_hyps_per_beam * 2 in case every hyps is terminated by EOS id.
    # Top K with eos removed.
    non_eos_mask = tf.not_equal(token_ids_for_beam, eos_id)
    num_candidate_hyps = num_hyps_per_beam * 2 * num_beams
    index = tf.where(
        non_eos_mask,
        tf.reshape(
            tf.range(num_candidate_hyps, dtype=tf.int32),
            token_scores_for_beam_shape),
        num_candidate_hyps *
        tf.ones(dtype=tf.int32, shape=token_scores_for_beam_shape))

    # Unrolled TopK.
    sorted_indices = []
    # Finds the first num_hyps_per_beam unmasked indexes and stores them in
    # concated_index (shape: [num_beams, num_candidate_hyps])
    # This is done by iteratively record the min index in each row, and reset
    # it to the max, so that next iteration reduce_min returns the 2nd minimum
    # index.
    for _ in range(num_hyps_per_beam):
      min_index = tf.reshape(tf.reduce_min(index, [1]), [num_beams, 1])
      sorted_indices.append(min_index)
      # Replace position with num_candidate_hyps value.
      index = tf.where(
          tf.equal(index, min_index),
          num_candidate_hyps *
          tf.ones(dtype=tf.int32, shape=token_scores_for_beam_shape), index)

    # Post processing ops to output expected tensors.
    concated_sorted_indices = tf.concat(sorted_indices, 1)
    flat_sorted_indices = tf.reshape(concated_sorted_indices, [-1])

    # A tensor of shape [b * k] with scores of the token selected.
    out_scores = tf.reshape(
        fast_gather(
            tf.reshape(token_scores_for_beam, [-1, 1]), flat_sorted_indices,
            num_candidate_hyps), [-1, 1])
    out_scores = tf.reshape(
        reorder_tensor("div_to_mod", out_scores, num_beams, num_hyps_per_beam),
        [-1])

    # Gather the updated histories of selected hypotheses if path merging is
    # enabled. Otherwise, the histories are unused, so just output in_histories.
    if merge_paths:
      flat_histories = tf.reshape(histories, [-1, 1])
      # [num_beams, 2 * num_hyps_per_beam]
      histories_for_beam = tf.reshape(
          fast_gather(flat_histories, flat_candidates_indices_in_top_k,
                      total_num_candidates), token_scores_for_beam_shape)
      out_histories = tf.reshape(
          fast_gather(
              tf.reshape(histories_for_beam, [-1, 1]), flat_sorted_indices,
              num_candidate_hyps), [-1, 1])
      out_histories = tf.reshape(
          reorder_tensor("div_to_mod", out_histories, num_beams,
                         num_hyps_per_beam), [-1])
    else:
      out_histories = in_histories

    prev_hyps_ids = tf.reshape(
        tf.reshape(
            fast_gather(
                tf.reshape(hyps_id, [1, -1]),
                flat_sorted_indices,
                num_candidate_hyps,
                max_value=num_hyps_per_beam,
                axis=1), [num_beams, -1]) * num_beams +
        tf.expand_dims(tf.range(num_beams), 1), [-1, 1])

    prev_hyps_ids = reorder_tensor(
        "div_to_mod",
        prev_hyps_ids,
        num_beams,
        num_hyps_per_beam,
        max_value=num_hyps_per_beam)
    # A tensor of shape [b * k] with index to the previous hyps which was
    # selected.
    out_prev_hyps = tf.reshape(prev_hyps_ids, [-1])

    # A tensor of shape [b * k, seq_len] which contain the attention
    # probabilities over the source words against word in the previous hyps.
    out_atten_probs = tf.reshape(
        fast_gather(in_atten_probs, out_prev_hyps,
                    num_beams * num_hyps_per_beam),
        [num_beams * num_hyps_per_beam, -1])

    sorted_top_k_ids = fast_gather(
        tf.reshape(token_ids_for_beam, [1, -1]),
        flat_sorted_indices,
        num_candidate_hyps,
        max_value=num_classes - 1,
        axis=1)
    sorted_top_k_ids = reorder_tensor(
        "div_to_mod",
        sorted_top_k_ids,
        num_beams,
        num_hyps_per_beam,
        max_value=num_classes - 1,
        axis=1)

    # A tensor of shape [b * k] with ids of the token selected.
    out_hyps = tf.reshape(sorted_top_k_ids, [-1])

    # A tensor of shape [b * k]. The cumulative score of the selected hyps after
    # the current decoding step.
    out_cumulative_scores = tf.reshape(
        fast_gather(
            tf.reshape(global_scores_for_beam, [-1, 1]), flat_sorted_indices,
            num_candidate_hyps), [-1, 1])

    out_cumulative_scores = tf.reshape(
        reorder_tensor("div_to_mod", out_cumulative_scores, num_beams,
                       num_hyps_per_beam), [-1])
    out_best_scores = tf.maximum(best_scores, in_best_scores)

    # A scalar, whether decoding should terminate for all beams.
    out_all_done = tf.reshape(
        tf.math.logical_not(
            tf.reduce_any(
                tf.greater(
                    out_cumulative_scores,
                    tf.reshape(
                        tf.tile(
                            tf.reshape(out_best_scores - beam_size, [-1, 1]),
                            [1, num_hyps_per_beam]), [-1])))), [])

    return (out_best_scores, out_cumulative_scores, out_scores, out_eos_scores,
            out_hyps, out_prev_hyps, out_done_hyps, out_atten_probs,
            eos_atten_probs, out_all_done, out_histories)
