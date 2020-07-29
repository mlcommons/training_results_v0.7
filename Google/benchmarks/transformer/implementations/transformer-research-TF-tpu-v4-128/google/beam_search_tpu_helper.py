# Lint as: python2, python3
"""Helper class for implementing a beam search decoder on TPUs.

Individual models just need to provide a few callback functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo import compat as tf
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import base_layer
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import beam_search_helper
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import py_utils
from six.moves import range
from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.lingvo.core import ops
from REDACTED.tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import

from REDACTED.tensorflow_models.mlperf.models.rough.transformer_lingvo.google.ops import beam_search_tpu_ops


class BeamSearchTpuHelper(base_layer.BaseLayer):
  """Helper class for performing beam search.

  The user of this helper class needs to implement three callbacks.

  def InitBeamSearchState(encoder_outputs, num_hyps_per_beam):
    Args:
      theta: A NestedMap object containing weights' values of this
          layer and its children layers.
      encoder_outputs: A NestedMap computed by encoder.
      num_hyps_per_beam: An int, number hyps to keep for source sentence.
    Returns:
      initial_results: a NestedMap of initial results. It should contain the
      following tensors at the minimum.
          log_probs: The initial log probs for each of the tokens in
              the target vocab, of shape [tgt_batch, vocab_size].
          atten_probs: The initial attention probs, of shape [tgt_batch,
              src_len].
      states: a NestedMap of tensors representing states that the client would
      like to keep track of for each hyp.
  This callback is called once only at the beginning of beam search.

  def PreBeamSearchStepCallback(
      encoder_outputs, step_ids, in_states, num_hyps_per_beam):
    Args:
      theta: A NestedMap object containing weights' values of this
          layer and its children layers.
      encoder_outputs: A NestedMap computed by encoder.
      step_ids: A tensor of shape [tgt_batch, 1].
      in_states: A NestedMap of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
    Returns:
      results: A NestedMap of beam search results. It should contain
          the 'atten_probs' and 'log_probs' tensors at the minimal. Optionally
          it may contain 'lm_log_probs' if a LM is used during decoding and
          'is_last_chunk' if it is decoding a neural transducer model.

          atten_probs: The updated attention probs, of shape [tgt_batch,
              src_len].
          log_probs: Log prob for each of the tokens in the target vocab. This
              is of shape [tgt_batch, vocab_size].
          lm_log_probs: Language model prob for each of the tokens in the target
              vocab.  If not empty, this is of shape [tgt_batch, vocab_size].
          is_last_chunk: Whether or not each of the hyp is at the end of a
          chunk. If non-empty, it is of shape [tgt_batch, 1]
      out_states: A NestedMap. The updated states. This 'out_states' should be
          of the exact same structure as 'in_states'
  This callback is called once every decoding time step before beam_search_step
  is called.

  def PostBeamSearchStepCallback(
      encoder_outputs, new_step_ids, other_states):
    Args:
      theta: A NestedMap object containing weights' values of this
          layer and its children layers.
      encoder_outputs: A NestedMap computed by encoder.
      new_step_ids: Token ids for the next beam search step.
      other_states: A NestedMap.
    Returns:
      final_states, A NestedMap.
  This callback is called once every decoding time step after beam_search_step
  is called.
  """

  @classmethod
  def Params(cls):
    p = super(BeamSearchTpuHelper, cls).Params()
    p.Define('num_hyps_per_beam', 8,
             'Num of hyps to keep per beam during decoding.')
    p.Define(
        'target_seq_length_ratio', 1.0,
        'Ratio of the average target sequence length over the average '
        'source sequence length.')
    p.Define('length_normalization', 0.0,
             'Beam search length normalization ratio.')
    p.Define('coverage_penalty', 0.0, 'Beam search coverage penalty.')
    p.Define(
        'valid_eos_max_logit_delta', 5.0,
        'During beam search, allow </s> to terminate a hyp only if its '
        'logit is no more than than this value away from the logit of the '
        'best candidate.')
    p.Define(
        'local_eos_threshold', -100.0,
        'During beam search, allow </s> to terminate a hyp if the local score '
        'for </s> is greater than local_eos_threshold.')
    p.Define(
        'beam_size', 3.0,
        'The maximum difference between best hyp and the worst in a beam.'
        ' This allows to prune our search when none of the active hyp is'
        ' close enough to the current best.')
    p.Define('target_sos_id', 1, 'Id of the start of sentence token.')
    p.Define('target_eos_id', 2, 'Id of the end of sentence token.')
    p.Define('target_seq_len', 0, 'Maximum allowed target seq length.')
    p.Define(
        'target_eoc_id', -1,
        'Id of the end of chunk token. Used by neural transducer only.'
        ' Set this id to a non-negative value only for NT.')
    p.Define(
        'merge_paths', False, 'If true, hyps which are identical when '
        'epsilons are removed will be combined into a single hyp.  The '
        'probability for that combined hyp will be the sum of the '
        'probabilities of the component hyps.  This can only be applied '
        'for epsilon-emitting models (RNN-T and NT).')
    p.Define(
        'batch_major_state', True, 'If True, we use batch as the major '
        'dimension of the hyp states. Otherwise, timing becomes the major '
        'dimension, and the gathers are performed along the second-to-major '
        'dimension.')
    p.Define(
        'batch_major_compute', False, 'If True, the target batch dimension '
        'is organized as num_beams by num_hyps_per_beam during the '
        'ExtendStep computation and the cache is stored following this order. '
        'So the topk indices into the cache for ReOrderHyps needs to be '
        'reordered before usage. Otherwise, the indices will be directly used '
        'without extra transformation.')
    p.Define(
        'short_seq_limit', 0,
        'An integer, the sequence length limit for using early stop '
        'method in attention layer (batch-major implementation). The sequence '
        'is always treated as the default long sequence for decoding when the '
        'limit is set to 0. For typical mt transformer config '
        '(batch 16, sequence length 150), the break even point is around 40 '
        'on REDACTED, and 50 on REDACTED. This may slightly change for '
        'different batch size and sequence length, which requires more '
        'experiments to set the value.')
    p.name = 'tpu_beam_search'
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BeamSearchTpuHelper, self).__init__(params)

  def _BeamSearchStep(self,
                      theta,
                      encoder_outputs,
                      cur_step,
                      step_ids,
                      core_bs_states,
                      other_states,
                      num_hyps_per_beam,
                      pre_beam_search_step_callback,
                      post_beam_search_step_callback,
                      use_short_seq_opt=False):
    """Extend beam search hyps for one step.

    num_beams = Number of source sequences to be decoded.
    num_hyps_per_beam = Number of hyps to keep per source sequence.
    num_hyps = num_beams * num_hyps_per_beam
    src_seq_len = Number of time steps in the source sequence.
    tgt_seq_len = Maximum allowed time steps in the target sequence.

    Args:
      theta: A NestedMap object containing weights' values of the decoder
        layer and its children layers.
      encoder_outputs: A NestedMap computed by encoder.
      cur_step: A scalar int tensor, the current time step, 0-based.
      step_ids: An int tensor of shape [num_hyps, 1]. The input ids to the
          current search step.
      core_bs_states: A tuple of core beam search states. This list is
          maintained by this helper class.
      other_states: A NestedMap of other beam search states. This NestedMap is
          managed and updated by the client. It is expected that each of its
          member tensors are of rank >= 1. t[i, ...] is the state of the i-th
          hyp at the beginning of this search step.
      num_hyps_per_beam: Num of hyps to keep per beam.
      pre_beam_search_step_callback: The PreBeamSearchStepCallback callback.
        Please refer to the class header comments for more details.
      post_beam_search_step_callback: The PostBeamSearchStepCallback callback.
        Please refer to the class header comments for more details.
      use_short_seq_opt: A bool, whether using short sequence optimization.
    Returns:
      A tuple of following elements for the next beam search step:
      (next step, all_done, step_ids, core_bs_states, other_states)
    """
    p = self.params

    if use_short_seq_opt:
      bs_results, other_states = pre_beam_search_step_callback(
          theta, encoder_outputs, step_ids, other_states, num_hyps_per_beam,
          use_short_seq_opt)
    else:
      bs_results, other_states = pre_beam_search_step_callback(
          theta, encoder_outputs, step_ids, other_states, num_hyps_per_beam)

    (best_scores, cumulative_scores, histories, in_scores, in_hyps,
     in_prev_hyps, in_done_hyps, in_atten_probs, in_eos_scores,
     in_eos_atten_probs) = core_bs_states

    (out_best_scores, out_cumulative_scores, out_scores, out_eos_scores,
     out_hyps, out_prev_hyps, out_done_hyps, out_atten_probs,
     out_eos_atten_probs, all_done,
     out_histories) = beam_search_tpu_ops.beam_search_step(
         bs_results.log_probs,
         bs_results.atten_probs,
         best_scores,
         cumulative_scores,
         histories,
         cur_step,
         eos_id=p.target_eos_id,
         beam_size=p.beam_size,
         num_beams=tf.shape(best_scores)[0],
         num_hyps_per_beam=num_hyps_per_beam,
         valid_eos_max_logit_delta=p.valid_eos_max_logit_delta,
         merge_paths=p.merge_paths,
         eoc_id=p.target_eoc_id if p.merge_paths else -1,
         is_last_chunk=bs_results.get('is_last_chunk'))

    # Write out values into TensorArray's corresponding to each output.
    arr_scores = in_scores.write(cur_step, out_scores)
    arr_eos_scores = in_eos_scores.write(cur_step, out_eos_scores)
    arr_hyps = in_hyps.write(cur_step, out_hyps)
    arr_prev_hyps = in_prev_hyps.write(cur_step, out_prev_hyps)
    # TODO(rohananil): Change the implementation of TensorArray write for
    # tf.bool from false += current_value to logical_and(true, current_value) as
    # addition operator for bool is not defined.
    arr_done_hyps = in_done_hyps.write(cur_step, tf.cast(
        out_done_hyps, tf.int32))
    arr_atten_probs = in_atten_probs.write(cur_step, out_atten_probs)
    arr_eos_atten_probs = in_eos_atten_probs.write(cur_step,
                                                   out_eos_atten_probs)

    # New beam search states.
    new_bs_states = (out_best_scores, out_cumulative_scores, out_histories,
                     arr_scores, arr_hyps, arr_prev_hyps, arr_done_hyps,
                     arr_atten_probs, arr_eos_scores, arr_eos_atten_probs)

    old_hyp_ids = tf.reshape(out_prev_hyps, [-1])

    if p.batch_major_compute:
      # Transformed the indices into the key/value cache for fast decoding
      # (prefix_states in other_states) due to the num_hyps dimension of
      # cache is computed as num_beams by num_hyps_per_beam, which is different
      # from the old_hyp_ids assumption (num_hyps_per_beam by num_beams).
      # Both transpose and recomputation are required to correct the indices.
      num_beams = tf.shape(best_scores)[0]
      old_hyp_ids_in_cache_order = tf.reshape(
          tf.transpose(tf.reshape(old_hyp_ids, [num_hyps_per_beam, -1])), [-1])
      old_hyp_ids_in_cache_order = (
          (old_hyp_ids_in_cache_order % num_beams) * num_hyps_per_beam +
          old_hyp_ids_in_cache_order // num_beams)

    def ReOrderHyps(x_in):
      """Reorders x_in based on prev hyp ids."""
      if isinstance(x_in, tf.Tensor) and x_in.shape.ndims > 0:
        # For rank > 1 tensors we make use of an efficient matmul based gather
        # on tpu that takes in account the range of the values. For R1, we
        # rely on the tf.gather and xla to optimize it efficiently for R1
        # layout.
        if x_in.shape.ndims > 1:
          if p.batch_major_state:
            num_hyps = tf.shape(old_hyp_ids)[0]
            x_out = beam_search_tpu_ops.fast_gather(
                x_in,
                old_hyp_ids,
                num_hyps,
                max_value=None,
                batch_major_state=p.batch_major_state)
          else:
            # Use corrected indices only here for batch major compute as
            # key/value caches are the states being affected.
            correct_old_hyp_ids = (
                old_hyp_ids_in_cache_order
                if p.batch_major_compute else old_hyp_ids)

            def _GatherStep(x_in, t):
              """Gather for one time step.

              Args:
                x_in: in the shape of [T, B, ...] we first get slice(t) from the
                  tensors, then gather old_hyp_ids from the slice and write the
                  interpolated slice inplace to update the original x_in.
                t: current time step

              Returns:
                Updated x_in and time step
              """
              x = tf.gather(tf.gather(x_in, t), correct_old_hyp_ids)
              return inplace_ops.alias_inplace_update(x_in, t, x), t + 1

            x_out, _ = tf.while_loop(lambda _, t: t <= cur_step, _GatherStep,
                                     (x_in, tf.zeros([], tf.int32)))
        else:
          x_out = tf.gather(x_in, old_hyp_ids)
        x_out.set_shape(x_in.get_shape())
        return x_out
      else:
        return x_in

    new_other_states = other_states.Transform(ReOrderHyps)
    new_step_ids = tf.reshape(out_hyps, [-1, 1])
    final_other_states = post_beam_search_step_callback(
        theta, encoder_outputs, new_step_ids, new_other_states)

    return (cur_step + 1, all_done, new_step_ids, new_bs_states,
            final_other_states)

  def _BeamSearchDecodeIds(self,
                           theta,
                           encoder_outputs,
                           num_hyps_per_beam,
                           init_beam_search_state=None,
                           pre_beam_search_step_callback=None,
                           post_beam_search_step_callback=None,
                           max_steps=None):
    """Performs beam-search based decoding.

    Args:
      theta: A NestedMap object containing weights' values of the decoder layer
        and its children layers.
      encoder_outputs: A NestedMap computed by encoder.
      num_hyps_per_beam: Number of hyps per beam.

      init_beam_search_state: The InitBeamSearchState callback. Please refer to
          the class header comments for more details.
      pre_beam_search_step_callback: The PreBeamSearchStepCallback callback.
          Please refer to the class header comments for more details.
      post_beam_search_step_callback: The PostBeamSearchStepCallback callback.
          Please refer to the class header comments for more details.
      max_steps: maximum beam search steps. If None, use
          self.params.target_seq_len.

    Returns:
      hyps: A tensor of shape [time, b * k] with ids of the token selected.
      prev_hyps: A tensor of shape [time, b * k] with index to the previous hyps
        which was selected.
      done_hyps: A boolean tensor of shape [time, b * k] where value
        indicates if hyps was terminated.
      scores: A tensor of shape [time, b * k] with scores of the token
        selected.
      atten_probs: A tensor of shape [time, b * k, seq_len] which contain the
        attention probabilities over the source words against word in the
        previous hyps.
      eos_scores: A tensor of shape [time, b * k] with scores of the eos token
        selected.
      eos_atten_probs: A tensor of shape [time, b * k, seq_len] which contain
        the attention probabilities over the source words against word in the
        previous hyps.
      source_seq_lengths:  A tensor of shape [time] containing the source
        seq_lengths.
      flat_final_other_states: A array of tensors that are part of other states.
    """
    p = self.params
    source_paddings = encoder_outputs.padding

    initial_results, other_states = init_beam_search_state(
        theta, encoder_outputs, num_hyps_per_beam)

    num_hyps = tf.shape(initial_results.log_probs)[0]
    num_beams = num_hyps // num_hyps_per_beam

    # We cache the NestedMap as member variable so that we can use it to
    # pack the final outputs. Tpu rewrite methods forces us to strictly pass
    # in Tensors, and output Tensors
    self._other_states = other_states

    step_ids = tf.fill([num_hyps, 1],
                       tf.constant(p.target_sos_id, dtype=tf.int32))
    min_score = -1e36
    fprop_dtype = py_utils.FPropDtype(p)
    best_scores = (tf.zeros(shape=[num_beams], dtype=fprop_dtype) + min_score)
    cumulative_scores = tf.zeros(shape=[num_hyps], dtype=fprop_dtype)
    histories = tf.zeros(shape=[num_hyps], dtype=tf.int32)
    in_scores = tf.TensorArray(dtype=fprop_dtype, size=max_steps)
    in_hyps = tf.TensorArray(dtype=tf.int32, size=max_steps)
    in_prev_hyps = tf.TensorArray(dtype=tf.int32, size=max_steps)
    in_done_hyps = tf.TensorArray(dtype=tf.int32, size=max_steps)
    in_atten_probs = tf.TensorArray(dtype=fprop_dtype, size=max_steps)
    in_eos_scores = tf.TensorArray(dtype=fprop_dtype, size=max_steps)
    in_eos_atten_probs = tf.TensorArray(dtype=fprop_dtype, size=max_steps)
    cur_step = tf.constant(0, dtype=tf.int32)
    all_done = tf.constant(False, dtype=tf.bool)
    # States for beam search that are inputs into Beam search step.
    accum_bs_states = [best_scores, cumulative_scores, histories]
    # States that are not accumulators.
    non_accum_bs_states = [
        in_scores,
        in_hyps,
        in_prev_hyps,
        in_done_hyps,
        in_atten_probs,
        in_eos_scores,
        in_eos_atten_probs,
    ]
    core_bs_states = tuple(accum_bs_states + non_accum_bs_states)

    flat_other_states = other_states.Flatten()

    # If there is an optimized implementation for short sequence, LoopBodyShort
    # will run first for short_seq_limit steps (after which the
    # LoopBodyShort does not have performance benefit). Then LoopBodyLong (the
    # default implementation) is used to continue the rest of the steps. For
    # decoders which do not have the short sequence specific implementation,
    # only the LoopBodyLong (the default implementation) will run.

    if p.short_seq_limit > 0:

      def LoopContinueShort(cur_step, all_done, unused_step_ids,
                            unused_core_bs_states, unused_other_states_list):
        """Use short_seq optimization when cur_step is smaller than limit."""
        return tf.math.logical_and(cur_step < p.short_seq_limit,
                                   tf.math.logical_not(all_done))

      def LoopBodyShort(cur_step, unused_all_done, step_ids, core_bs_states,
                        other_states_list):
        """Loop body of short_seq optimization.

        Instead of doing computation for the entire padded sequence, while loop
        with early exit is used within each _BeamSearchStep to do computation
        for only the actual sequence (seq_length <= cur_step).
        use_short_seq_opt is used as the flag to pass this information down to
        the decoder implementation.

        Args:
          cur_step: A scalar int tensor, the current time step, 0-based.
          unused_all_done: A tf.bool, indicating whether the decoding finishes.
          step_ids: An int32 tensor of shape [num_hyps, 1]. The input ids to the
            current search step.
          core_bs_states: A tuple of core beam search states.
          other_states_list: A flattened NestedMap of other beam search states.

        Returns:
          The updated input tuple, with the same shape.
        """
        (cur_step, all_done, new_step_ids, new_bs_states,
         new_other_states) = self._BeamSearchStep(
             theta,
             encoder_outputs,
             cur_step,
             step_ids,
             core_bs_states,
             other_states.Pack(other_states_list),
             num_hyps_per_beam,
             pre_beam_search_step_callback,
             post_beam_search_step_callback,
             use_short_seq_opt=True)
        return (cur_step, all_done, new_step_ids, new_bs_states,
                new_other_states.Flatten())

      (cur_step, all_done, step_ids, core_bs_states,
       flat_other_states) = tf.while_loop(
           LoopContinueShort,
           LoopBodyShort,
           loop_vars=(cur_step, all_done, step_ids, core_bs_states,
                      flat_other_states),
           parallel_iterations=10,
           back_prop=False,
           swap_memory=False,
           shape_invariants=(
               tf.TensorShape(cur_step.get_shape()),
               tf.TensorShape(all_done.get_shape()),
               tf.TensorShape(step_ids.get_shape()),
               tuple(
                   list(_GetShapes(accum_bs_states)) +
                   list(_GetShapes(non_accum_bs_states, none_shapes=True))),
               _GetShapes(flat_other_states, none_shapes=True)),
           maximum_iterations=max_steps)

    def LoopContinueLong(cur_step, all_done, unused_step_ids,
                         unused_core_bs_states, unused_other_states_list):
      """Continue default implementation until decoding finishes."""
      return tf.math.logical_and(cur_step < max_steps,
                                 tf.math.logical_not(all_done))

    def LoopBodyLong(cur_step, unused_all_done, step_ids, core_bs_states,
                     other_states_list):
      """Loop body of default long_seq implementation."""
      (cur_step, all_done, new_step_ids, new_bs_states,
       new_other_states) = self._BeamSearchStep(
           theta,
           encoder_outputs,
           cur_step,
           step_ids,
           core_bs_states,
           other_states.Pack(other_states_list),
           num_hyps_per_beam,
           pre_beam_search_step_callback,
           post_beam_search_step_callback,
           use_short_seq_opt=False)
      return (cur_step, all_done, new_step_ids, new_bs_states,
              new_other_states.Flatten())

    _, _, _, final_bs_states, flat_final_other_states = tf.while_loop(
        LoopContinueLong,
        LoopBodyLong,
        loop_vars=(cur_step, all_done, step_ids, core_bs_states,
                   flat_other_states),
        parallel_iterations=10,
        back_prop=False,
        swap_memory=False,
        shape_invariants=(
            tf.TensorShape(cur_step.get_shape()),
            tf.TensorShape(all_done.get_shape()),
            tf.TensorShape(step_ids.get_shape()),
            tuple(
                list(_GetShapes(accum_bs_states)) +
                list(_GetShapes(non_accum_bs_states, none_shapes=True))),
            _GetShapes(flat_other_states, none_shapes=False)),
        maximum_iterations=max_steps)

    if isinstance(source_paddings, py_utils.NestedMap):
      source_seq_lengths = tf.cast(
          tf.round(
              tf.reduce_sum(1.0 - tf.transpose(source_paddings.Flatten()[0]),
                            1)),
          dtype=tf.int32)
    else:
      source_seq_lengths = tf.cast(
          tf.round(tf.reduce_sum(1.0 - tf.transpose(source_paddings), 1)),
          dtype=tf.int32)

    # Concatenate all outputs on axis=0.
    scores = final_bs_states[3].stack()
    hyps = final_bs_states[4].stack()
    prev_hyps = final_bs_states[5].stack()
    done_hyps = tf.cast(final_bs_states[6].stack(), tf.bool)
    atten_probs = final_bs_states[7].stack()
    eos_scores = final_bs_states[8].stack()
    eos_atten_probs = final_bs_states[9].stack()
    rets = (hyps, prev_hyps, done_hyps, scores, atten_probs, eos_scores,
            eos_atten_probs, source_seq_lengths)

    # TODO(rohananil): Only send a single R1 tensor to host instead of 3 after
    # b/111131551 is resolved.
    # Canonical shapes for tensors of various. ranks
    r_shapes = [
        py_utils.GetShape(source_seq_lengths),
        py_utils.GetShape(hyps),
        py_utils.GetShape(atten_probs)
    ]
    # Reshape all tensors to [-1] to avoid cost of copy due to padding.
    rets_r1 = [tf.reshape(r, [-1]) for r in rets]

    return tuple(r_shapes) + tuple(rets_r1) + tuple(flat_final_other_states)

  def BeamSearchDecodePostProcess(self, num_hyps_per_beam, max_steps, r1_shape,
                                  r2_shape, r3_shape, hyps, prev_hyps,
                                  done_hyps, scores, atten_probs, eos_scores,
                                  eos_atten_probs, source_seq_lengths,
                                  *flat_final_other_states):
    """Beam search post processing functions on CPUs.


    Args:
      num_hyps_per_beam: Number of hyps per beam.
      max_steps: Maximum number of beam search steps.
      r1_shape: A tensor of shape [1] with value [time].
      r2_shape: A tensor of shape [2] with values [time, b * k].
      r3_shape: A tensor of shape [3] with values [time, b * k, seq_len].
      hyps: A tensor of shape [1] with ids of the token selected.
      prev_hyps: A tensor of shape [time * b * k] with index to the previous
        hyps which was selected.
      done_hyps: A boolean tensor of shape [time * b * k] where value
        indicates if hyps was terminated.
      scores: A tensor of shape [time * b * k] with scores of the token
        selected.
      atten_probs: A tensor of shape [time * b * k, seq_len] which contain the
        attention probabilities over the source words against word in the
        previous hyps.
      eos_scores: A tensor of shape [time * b * k] with scores of the eos token
        selected.
      eos_atten_probs: A tensor of shape [time * b * k, seq_len] which contain
        the attention probabilities over the source words against word in the
        previous hyps.
      source_seq_lengths:  A tensor of shape [time] containing the source
        seq_lengths.
      *flat_final_other_states: A array of tensors that are part of other
        states.

    Returns:
      final_done_hyps: A tensor of shape [time, b * k] containing `Hypothesis`
        pbs containing terminated hyps.
      topk_hyps, topk_ids, topk_lens, topk_scores: Top K terminated Hyps.
      flat_final_other_states: A array of tensors that are part of other states.
    """
    p = self.params

    def _ReshapeBackToHigherRank(inps, r_shape):
      for i in range(len(inps)):
        inps[i] = tf.reshape(inps[i], r_shape)
      return inps

    # Reshape all tensors back to original shapes of rank 1, 2 and 3.
    r1_inps = [source_seq_lengths]
    r1_inps = _ReshapeBackToHigherRank(r1_inps, r1_shape)
    r2_inps = [hyps, prev_hyps, done_hyps, scores, eos_scores]
    r2_inps = _ReshapeBackToHigherRank(r2_inps, r2_shape)
    r3_inps = [atten_probs, eos_atten_probs]
    r3_inps = _ReshapeBackToHigherRank(r3_inps, r3_shape)

    (source_seq_lengths, hyps, prev_hyps, done_hyps, scores, eos_scores,
     atten_probs, eos_atten_probs) = (
         r1_inps + r2_inps + r3_inps)

    final_done_hyps = ops.hyps_from_beam_search_outs(
        hyps,
        prev_hyps,
        done_hyps,
        scores,
        atten_probs,
        eos_scores,
        eos_atten_probs,
        eos_id=p.target_eos_id,
        num_hyps_per_beam=num_hyps_per_beam)
    topk_hyps = ops.top_k_terminated_hyps(
        final_done_hyps,
        source_seq_lengths,
        k=num_hyps_per_beam,
        num_hyps_per_beam=num_hyps_per_beam,
        length_normalization=p.length_normalization,
        coverage_penalty=p.coverage_penalty,
        target_seq_length_ratio=p.target_seq_length_ratio,
        eoc_id=p.target_eoc_id,
        merge_paths=p.merge_paths)
    topk_ids, topk_lens, topk_scores = ops.unpack_hyp(
        topk_hyps, max_seq_length=max_steps)
    # [num_beams, num_hyps_per_beam].
    topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))
    return (final_done_hyps, topk_hyps, topk_ids, topk_lens,
            topk_scores) + tuple(flat_final_other_states)

  def BeamSearchDecode(self,
                       theta,
                       encoder_outputs,
                       num_hyps_per_beam_override=0,
                       init_beam_search_state=None,
                       pre_beam_search_step_callback=None,
                       post_beam_search_step_callback=None,
                       max_steps=None,
                       normalize_output_ids=False):
    """Performs beam-search based decoding.

    Args:
      theta: A NestedMap object containing weights' values of the decoder layer
        and its children layers.
      encoder_outputs: A NestedMap computed by encoder.
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to override
        p.num_hyps_per_beam.
      init_beam_search_state: The InitBeamSearchState callback. Please refer to
        the class header comments for more details.
      pre_beam_search_step_callback: The PreBeamSearchStepCallback callback.
        Please refer to the class header comments for more details.
      post_beam_search_step_callback: The PostBeamSearchStepCallback callback.
        Please refer to the class header comments for more details.
      max_steps: maximum beam search steps, an int. If None, use
        self.params.target_seq_len.
      normalize_output_ids: A Python bool. If true, normalize the output with
        EOC ids removed from topk_ids and topk_lens updated to include the first
        EOS id per sequence. This basically include one extra step as calling
        `decoder_utils.NormalizeBeamSearchOutputIds`.

    Returns:
      beam_search_helper.BeamSearchDecodeOutput
    """
    p = self.params
    num_hyps_per_beam = self.params.num_hyps_per_beam
    if num_hyps_per_beam_override > 0:
      num_hyps_per_beam = num_hyps_per_beam_override
    max_steps = max_steps or p.target_seq_len

    rets_decode = self._BeamSearchDecodeIds(
        theta, encoder_outputs, num_hyps_per_beam, init_beam_search_state,
        pre_beam_search_step_callback, post_beam_search_step_callback,
        max_steps)
    return rets_decode


def _GetShapes(tensors, none_shapes=False):
  """Util for getting nested strucutre of shapes from structure of tensors.

  Args:
    tensors: Structure of Tensors to get shapes for.
    none_shapes: Returns None shapes if true.

  Returns:
    The same structure as tensors but of corresponding TensorShape objects.
  """
  shapes = []
  for t in tf.nest.flatten(tensors):
    shape = t.get_shape() if isinstance(t, tf.Tensor) else None
    if none_shapes:
      if shape:
        shapes.append(tf.TensorShape([None] * len(shape)))
      else:
        shapes.append(tf.TensorShape(None))
    else:
      shapes.append(tf.TensorShape(shape))

  return type(tensors)(tf.nest.pack_sequence_as(tensors, shapes))
