# Copyright 2018 Google LLC
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

"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

from absl import flags
import functools
import json
import logging
import os.path
import struct
import tempfile
import time
import numpy as np
import random

import tensorflow as tf
from tensorflow.distribute import cluster_resolver as contrib_cluster_resolver
from tensorflow.quantization import quantize as contrib_quantize
from tensorflow import summary as contrib_summary
#from tensorflow import tpu as contrib_tpu
from tensorflow.compat.v1.estimator import tpu as contrib_tpu_python_tpu_tpu_config
from tensorflow.compat.v1.estimator import tpu as contrib_tpu_python_tpu_tpu_estimator
#from tensorflow.tpu.python.tpu import tpu_optimizer as contrib_tpu_python_tpu_tpu_optimizer

import features as features_lib
import go
import symmetries
import minigo_model

import horovod.tensorflow as hvd
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
import intel_quantization.graph_converter as converter

from mlperf_logging import mllog

flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'dist_train', default=False,
    help=('Using distributed training or not.'))

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))

flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')

flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')

flags.DEFINE_string(
    'input_layout', 'nhwc',
    help='Layout of input features: "nhwc" or "nchw"')

flags.DEFINE_bool(
    'use_bfloat16', True,
    help='Choose whether to use bfloat16 in training.')

# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


class DualNetwork():
    def __init__(self, save_file):
        self.save_file = save_file
        self.inference_input = None
        self.inference_output = None
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        if FLAGS.gpu_device_list is not None:
            config.gpu_options.visible_device_list = FLAGS.gpu_device_list
        self.sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)
        self.initialize_graph()

    def initialize_graph(self):
        with self.sess.graph.as_default():
            features, labels = get_inference_input()
            params = FLAGS.flag_values_dict()
            logging.info('TPU inference is supported on C++ only. '
                         'DualNetwork will ignore use_tpu=True')
            params['use_tpu'] = False
            estimator_spec = model_fn(features, labels,
                                      tf.estimator.ModeKeys.PREDICT,
                                      params=params)
            self.inference_input = features
            self.inference_output = estimator_spec.predictions
            if self.save_file is not None:
                self.initialize_weights(self.save_file)
            else:
                self.sess.run(tf.global_variables_initializer())

    def initialize_weights(self, save_file):
        """Initialize the weights from the given save_file.
        Assumes that the graph has been constructed, and the
        save_file contains weights that match the graph. Used
        to set the weights to a different version of the player
        without redifining the entire graph."""
        tf.compat.v1.train.Saver().restore(self.sess, save_file)

    def run(self, position):
        probs, values = self.run_many([position])
        return probs[0], values[0]

    def run_many(self, positions):
        f = get_features()
        processed = [features_lib.extract_features(p, f) for p in positions]
        if FLAGS.use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        outputs = self.sess.run(self.inference_output,
                                feed_dict={self.inference_input: processed})
        probabilities, value = outputs['policy_output'], outputs['value_output']
        if FLAGS.use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)
        return probabilities, value


def get_features_planes():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES_PLANES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES_PLANES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)


def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)


def get_inference_input():
    """Set up placeholders for input features/labels.

    Returns the feature, output tensors that get passed into model_fn."""
    feature_type = tf.bool if FLAGS.bool_features else tf.float32
    if FLAGS.input_layout == 'nhwc':
        feature_shape = [None, go.N, go.N, get_features_planes()]
    elif FLAGS.input_layout == 'nchw':
        feature_shape = [None, get_features_planes(), go.N, go.N]
    else:
        raise ValueError('invalid input_layout "%s"' % FLAGS.input_layout)
    return (tf.compat.v1.placeholder(feature_type, feature_shape, name='pos_tensor'),
            {'pi_tensor': tf.compat.v1.placeholder(tf.float32, [None, go.N * go.N + 1]),
             'value_tensor': tf.compat.v1.placeholder(tf.float32, [None])})


def model_fn(features, labels, mode, params):
    """
    Create the model for estimator api

    Args:
        features: if input_layout == 'nhwc', a tensor with shape:
                [BATCH_SIZE, go.N, go.N, get_features_planes()]
            else, a tensor with shape:
                [BATCH_SIZE, get_features_planes(), go.N, go.N]
        labels: dict from string to tensor with shape
            'pi_tensor': [BATCH_SIZE, go.N * go.N + 1]
            'value_tensor': [BATCH_SIZE]
        mode: a tf.estimator.ModeKeys (batchnorm params update for TRAIN only)
        params: A dictionary (Typically derived from the FLAGS object.)
    Returns: tf.estimator.EstimatorSpec with props
        mode: same as mode arg
        predictions: dict of tensors
            'policy': [BATCH_SIZE, go.N * go.N + 1]
            'value': [BATCH_SIZE]
        loss: a single value tensor
        train_op: train op
        eval_metric_ops
    return dict of tensors
        logits: [BATCH_SIZE, go.N * go.N + 1]
    """
    if FLAGS.use_bfloat16:
        with tf.compat.v1.tpu.bfloat16_scope():
            policy_output, value_output, logits = model_inference_fn(
                features, mode == tf.estimator.ModeKeys.TRAIN, params)
    else:
        policy_output, value_output, logits = model_inference_fn(
            features, mode == tf.estimator.ModeKeys.TRAIN, params)

    # train ops
    policy_cost = tf.reduce_mean(
        tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=tf.stop_gradient(labels['pi_tensor'])))

    value_cost = params['value_cost_weight'] * tf.reduce_mean(
        tf.square(value_output - labels['value_tensor']))

    reg_vars = [v for v in tf.compat.v1.trainable_variables()
                if 'bias' not in v.name and 'beta' not in v.name]
    l2_cost = params['l2_strength'] * \
        tf.add_n([tf.compat.v1.nn.l2_loss(v) for v in reg_vars])

    combined_cost = policy_cost + value_cost + l2_cost

    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.piecewise_constant(
        global_step, params['lr_boundaries'], params['lr_rates'])
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

    # Insert quantization ops if requested
    if params['quantize']:
        if mode == tf.estimator.ModeKeys.TRAIN:
            contrib_quantize.create_training_graph(
                quant_delay=params['quant_delay'])
        else:
            contrib_quantize.create_eval_graph()

    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate, params['sgd_momentum'])
    if(params['dist_train']):
      optimizer = hvd.DistributedOptimizer(optimizer)
    #if params['use_tpu']:
    #    optimizer = contrib_tpu_python_tpu_tpu_optimizer.CrossShardOptimizer(
    #        optimizer)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(combined_cost, global_step=global_step)

    # Computations to be executed on CPU, outside of the main TPU queues.
    def eval_metrics_host_call_fn(policy_output, value_output, pi_tensor,
                                  value_tensor, policy_cost, value_cost,
                                  l2_cost, combined_cost, step,
                                  est_mode=tf.estimator.ModeKeys.TRAIN):
        policy_entropy = -tf.reduce_mean(tf.reduce_sum(
            policy_output * tf.compat.v1.log(policy_output), axis=1))
        # pi_tensor is one_hot when generated from sgfs (for supervised learning)
        # and soft-max when using self-play records. argmax normalizes the two.
        policy_target_top_1 = tf.argmax(pi_tensor, axis=1)

        policy_output_in_top1 = tf.compat.v1.to_float(
            tf.compat.v1.nn.in_top_k(policy_output, policy_target_top_1, k=1))
        policy_output_in_top3 = tf.compat.v1.to_float(
            tf.compat.v1.nn.in_top_k(policy_output, policy_target_top_1, k=3))

        policy_top_1_confidence = tf.reduce_max(policy_output, axis=1)
        policy_target_top_1_confidence = tf.boolean_mask(
            policy_output,
            tf.one_hot(policy_target_top_1, tf.shape(policy_output)[1]))

        value_cost_normalized = value_cost / params['value_cost_weight']
        avg_value_observed = tf.reduce_mean(value_tensor)

        with tf.compat.v1.variable_scope('metrics'):
            metric_ops = {
                'policy_cost': tf.compat.v1.metrics.mean(policy_cost),
                'value_cost': tf.compat.v1.metrics.mean(value_cost),
                'value_cost_normalized': tf.compat.v1.metrics.mean(value_cost_normalized),
                'l2_cost': tf.compat.v1.metrics.mean(l2_cost),
                'policy_entropy': tf.compat.v1.metrics.mean(policy_entropy),
                'combined_cost': tf.compat.v1.metrics.mean(combined_cost),
                'avg_value_observed': tf.compat.v1.metrics.mean(avg_value_observed),
                'policy_accuracy_top_1': tf.compat.v1.metrics.mean(policy_output_in_top1),
                'policy_accuracy_top_3': tf.compat.v1.metrics.mean(policy_output_in_top3),
                'policy_top_1_confidence': tf.compat.v1.metrics.mean(policy_top_1_confidence),
                'policy_target_top_1_confidence': tf.compat.v1.metrics.mean(
                    policy_target_top_1_confidence),
                'value_confidence': tf.compat.v1.metrics.mean(tf.abs(value_output)),
            }

        if est_mode == tf.estimator.ModeKeys.EVAL:
            return metric_ops

        # NOTE: global_step is rounded to a multiple of FLAGS.summary_steps.
        eval_step = tf.reduce_min(step)

        # Create summary ops so that they show up in SUMMARIES collection
        # That way, they get logged automatically during training
        summary_writer = contrib_summary.create_file_writer(FLAGS.work_dir)
        #with summary_writer.as_default(), \
        #        contrib_summary.record_summaries_every_n_global_steps(
        #            params['summary_steps'], eval_step):
        #    for metric_name, metric_op in metric_ops.items():
        #        contrib_summary.scalar(
        #            metric_name, metric_op[1], step=eval_step)

        # Reset metrics occasionally so that they are mean of recent batches.
        reset_op = tf.compat.v1.variables_initializer(tf.compat.v1.local_variables('metrics'))
        cond_reset_op = tf.cond(
            tf.equal(eval_step % params['summary_steps'], tf.compat.v1.to_int64(1)),
            lambda: reset_op,
            lambda: tf.no_op())

        #return contrib_summary.all_summary_ops() + [cond_reset_op]
        return [cond_reset_op]

    metric_args = [
        policy_output,
        value_output,
        labels['pi_tensor'],
        labels['value_tensor'],
        tf.reshape(policy_cost, [1]),
        tf.reshape(value_cost, [1]),
        tf.reshape(l2_cost, [1]),
        tf.reshape(combined_cost, [1]),
        tf.reshape(global_step, [1]),
    ]

    predictions = {
        'policy_output': policy_output,
        'value_output': value_output,
    }

    eval_metrics_only_fn = functools.partial(
        eval_metrics_host_call_fn, est_mode=tf.estimator.ModeKeys.EVAL)
    host_call_fn = functools.partial(
        eval_metrics_host_call_fn, est_mode=tf.estimator.ModeKeys.TRAIN)

    tpu_estimator_spec = contrib_tpu_python_tpu_tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=combined_cost,
        train_op=train_op,
        eval_metrics=(eval_metrics_only_fn, metric_args),
        host_call=(host_call_fn, metric_args)
    )
    if params['use_tpu']:
        return tpu_estimator_spec
    else:
        return tpu_estimator_spec.as_estimator_spec()


def model_inference_fn(features, training, params):
    """Builds just the inference part of the model graph.

    Args:
        features: input features tensor.
        training: True if the model is training.
        params: A dictionary

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """
    if FLAGS.bool_features:
        if FLAGS.use_bfloat16:
            features = tf.dtypes.cast(features, dtype=tf.bfloat16)
        else:
            features = tf.dtypes.cast(features, dtype=tf.float32)

    if FLAGS.input_layout == 'nhwc':
        bn_axis = -1
        data_format = 'channels_last'
    else:
        bn_axis = 1
        data_format = 'channels_first'

    mg_batchn = functools.partial(
        tf.compat.v1.layers.batch_normalization,
        axis=bn_axis,
        momentum=.95,
        epsilon=1e-5,
        center=True,
        scale=True,
        fused=True,
        training=training)

    mg_conv2d = functools.partial(
        tf.compat.v1.layers.conv2d,
        filters=params['conv_width'],
        kernel_size=3,
        padding='same',
        use_bias=False,
        data_format=data_format)

    mg_global_avgpool2d = functools.partial(
        tf.compat.v1.layers.average_pooling2d,
        pool_size=go.N,
        strides=1,
        padding='valid',
        data_format=data_format)

    def mg_activation(inputs):
        if FLAGS.use_swish:
            return tf.compat.v1.nn.swish(inputs)

        return tf.compat.v1.nn.relu(inputs)

    def residual_inner(inputs):
        conv_layer1 = mg_batchn(mg_conv2d(inputs))
        initial_output = mg_activation(conv_layer1)
        conv_layer2 = mg_batchn(mg_conv2d(initial_output))
        return conv_layer2

    def mg_res_layer(inputs):
        residual = residual_inner(inputs)
        output = mg_activation(inputs + residual)
        return output

    def mg_squeeze_excitation_layer(inputs):
        # Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
        # 2018 IEEE/CVF Conference on Computer Vision, 7132-7141.
        # arXiv:1709.01507 [cs.CV]

        channels = params['conv_width']
        ratio = FLAGS.SE_ratio
        assert channels % ratio == 0

        residual = residual_inner(inputs)
        pool = mg_global_avgpool2d(residual)
        fc1 = tf.compat.v1.layers.dense(pool, units=channels // ratio)
        squeeze = mg_activation(fc1)

        if FLAGS.use_SE_bias:
            fc2 = tf.compat.v1.layers.dense(squeeze, units=2*channels)
            # Channels_last so axis = 3 = -1
            gamma, bias = tf.split(fc2, 2, axis=3)
        else:
            gamma = tf.compat.v1.layers.dense(squeeze, units=channels)
            bias = 0

        sig = tf.compat.v1.nn.sigmoid(gamma)
        # Explicitly signal the broadcast.
        scale = tf.reshape(sig, [-1, 1, 1, channels])

        excitation = tf.multiply(scale, residual) + bias
        return mg_activation(inputs + excitation)

    initial_block = mg_activation(mg_batchn(mg_conv2d(features)))

    # the shared stack
    shared_output = initial_block
    for _ in range(params['trunk_layers']):
        if FLAGS.use_SE or FLAGS.use_SE_bias:
            shared_output = mg_squeeze_excitation_layer(shared_output)
        else:
            shared_output = mg_res_layer(shared_output)

    # Policy head
    policy_conv = mg_conv2d(
        shared_output, filters=params['policy_conv_width'], kernel_size=1)
    policy_conv = mg_activation(
        mg_batchn(policy_conv, center=False, scale=False))

    if FLAGS.input_layout == 'nhwc':
        policy_conv = tf.transpose(policy_conv, [0, 3, 1, 2])

    logits = tf.compat.v1.layers.dense(
        tf.reshape(
            policy_conv, [-1, params['policy_conv_width'] * go.N * go.N]),
        go.N * go.N + 1)

    if FLAGS.use_bfloat16:
        policy_output = tf.compat.v1.nn.softmax(tf.cast(logits, tf.float32), name='policy_output')
    else:
        policy_output = tf.compat.v1.nn.softmax(logits, name='policy_output')

    # Value head
    value_conv = mg_conv2d(
        shared_output, filters=params['value_conv_width'], kernel_size=1)
    value_conv = mg_activation(
        mg_batchn(value_conv, center=False, scale=False))

    if FLAGS.input_layout == 'nhwc':
        value_conv = tf.transpose(value_conv, [0, 3, 1, 2])

    value_fc_hidden = mg_activation(tf.compat.v1.layers.dense(
        tf.reshape(value_conv, [-1, params['value_conv_width'] * go.N * go.N]),
        params['fc_width']))
    if FLAGS.use_bfloat16:
        value_output = tf.compat.v1.nn.tanh(
            tf.cast(tf.reshape(tf.compat.v1.layers.dense(value_fc_hidden, 1), [-1]), tf.float32),
            name='value_output')
        logits = tf.cast(logits, tf.float32)
    else:
        value_output = tf.compat.v1.nn.tanh(
            tf.reshape(tf.compat.v1.layers.dense(value_fc_hidden, 1), [-1]),
            name='value_output')

    return policy_output, value_output, logits


def tpu_model_inference_fn(features):
    """Builds the model graph suitable for running on TPU.

    It does two things:
     1) Mark all weights as constant, which improves TPU inference performance
        because it prevents the weights being transferred to the TPU every call
        to Session.run().
     2) Adds constant to the graph with a unique value and marks it as a
        dependency on the rest of the model. This works around a TensorFlow bug
        that prevents multiple models being run on a single TPU.

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """
    def custom_getter(getter, name, *args, **kwargs):
        with tf.control_dependencies(None):
            return tf.guarantee_const(
                getter(name, *args, **kwargs), name=name + '/GuaranteeConst')
    with tf.compat.v1.variable_scope('', custom_getter=custom_getter):
        # TODO(tommadams): remove the tf.control_dependencies context manager
        # when a fixed version of TensorFlow is released.
        t = int(time.time())
        epoch_time = tf.constant(t, name='epoch_time_%d' % t)
        with tf.control_dependencies([epoch_time]):
            if FLAGS.input_layout == 'nhwc':
                feature_shape = [-1, go.N, go.N, get_features_planes()]
            else:
                feature_shape = [-1, get_features_planes(), go.N, go.N]
            features = tf.reshape(features, feature_shape)
            return model_inference_fn(features, False, FLAGS.flag_values_dict())


def maybe_set_seed():
    
    if FLAGS.training_seed != 0:
        random.seed(FLAGS.training_seed)
        tf.set_random_seed(FLAGS.training_seed)
        np.random.seed(FLAGS.training_seed)
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.SEED, value=FLAGS.training_seed) 


def get_estimator(num_intra_threads=0, num_inter_threads=0):
    if FLAGS.use_tpu:
        return _get_tpu_estimator()
    else:
        return _get_nontpu_estimator(num_intra_threads, num_inter_threads)


def _get_nontpu_estimator(num_intra_threads, num_inter_threads):
    session_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=num_intra_threads,
        inter_op_parallelism_threads=num_inter_threads)
    session_config.gpu_options.allow_growth = True
    model_dir = None
    if (not FLAGS.dist_train) or (hvd.rank()==0):
        model_dir = FLAGS.work_dir
        step_count_steps = 50
        summary_steps = FLAGS.summary_steps
    else:
        step_count_steps = 10000000
        summary_steps = 10000000

    run_config = tf.estimator.RunConfig(
        log_step_count_steps = step_count_steps,
        save_summary_steps=summary_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        session_config=session_config)
    return tf.estimator.Estimator(
        model_fn,
        model_dir=model_dir,
        config=run_config,
        params=FLAGS.flag_values_dict())


def _get_tpu_estimator():
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=None, project=None)
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    run_config = contrib_tpu_python_tpu_tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=FLAGS.work_dir,
        save_checkpoints_steps=max(1000, FLAGS.iterations_per_loop),
        save_summary_steps=FLAGS.summary_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        session_config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=contrib_tpu_python_tpu_tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=contrib_tpu_python_tpu_tpu_config.InputPipelineConfig.PER_HOST_V2))

    return contrib_tpu_python_tpu_tpu_estimator.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size * FLAGS.num_tpu_cores,
        eval_batch_size=FLAGS.train_batch_size * FLAGS.num_tpu_cores,
        params=FLAGS.flag_values_dict())


def bootstrap():
    """Initialize a tf.Estimator run with random initial weights."""
    # a bit hacky - forge an initial checkpoint with the name that subsequent
    # Estimator runs will expect to find.
    #
    # Estimator will do this automatically when you call train(), but calling
    # train() requires data, and I didn't feel like creating training data in
    # order to run the full train pipeline for 1 step.
    maybe_set_seed()
    initial_checkpoint_name = 'model.ckpt-1'
    save_file = os.path.join(FLAGS.work_dir, initial_checkpoint_name)
    sess = tf.compat.v1.Session(graph=tf.Graph())
    with sess.graph.as_default():
        features, labels = get_inference_input()
        model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                 params=FLAGS.flag_values_dict())
        sess.run(tf.global_variables_initializer())
        tf.compat.v1.train.Saver().save(sess, save_file)


def export_model(model_path):
    """Take the latest checkpoint and copy it to model_path.

    Assumes that all relevant model files are prefixed by the same name.
    (For example, foo.index, foo.meta and foo.data-00000-of-00001).

    Args:
        model_path: The path (can be a gs:// path) to export model
    """
    FLAGS.use_bfloat16 = False
    estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.work_dir,
                                       params=FLAGS.flag_values_dict())
    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.io.gfile.glob(latest_checkpoint + '*')
    mllogger = mllog.get_mllogger()
    mllog.config(filename="train.log")

    mllog.config(
      default_namespace = "worker1",
      default_stack_offset = 1,
      default_clear_line = False)

    for filename in all_checkpoint_files:
        suffix = filename.partition(latest_checkpoint)[2]
        destination_path = model_path + suffix
        logging.info('Copying {} to {}'.format(filename, destination_path))
        tf.io.gfile.copy(filename, destination_path)
        

def minigo_callback_cmds(tf_records):
  script = 'numactl -N 0 -l python3 produce_min_max_log.py'
  flags= ' --input_graph={}'
  flags += ' --flagfile={0}'.format(os.path.join(FLAGS.flags_dir, 'architecture.flags'))
  flags += ' --data_location={0}'.format(tf_records)
  flags += ' --num_steps={0}'.format(FLAGS.quantize_test_steps)
  flags += ' --batch_size={0}'.format(FLAGS.quantize_test_batch_size)
  flags += ' --random_rotation={0}'.format(FLAGS.random_rotation)
  logging.info(script + flags)
  return script + flags

def quantization(input_graph, model_path, tf_records):
  minigo_converter = converter.GraphConverter(
        input_graph,
        model_path + '.pb',
        ["pos_tensor"],
        ["policy_output", "value_output"])
  minigo_converter.gen_calib_data_cmds = minigo_callback_cmds(tf_records)
  minigo_converter.quantize()

def optimize_graph(input_graph, model_path, quantizing_graph, tf_records):
  fp32_graph = graph_pb2.GraphDef()
  with tf.compat.v1.gfile.Open(input_graph, "rb") as read_f:
      data = read_f.read()
      fp32_graph.ParseFromString(data)

  opt_graph = optimize_for_inference(
      fp32_graph,
      ["pos_tensor"],
      ["policy_output", "value_output"],
      dtypes.bool.as_datatype_enum,
      False)

  with tf.io.gfile.GFile(model_path + '.pb', 'wb') as write_f:
      write_f.write(opt_graph.SerializeToString())

  if(quantizing_graph):
    fp32_opt_graph = model_path + '.pb'
    output_graph = quantization(fp32_opt_graph, model_path, tf_records)

def get_input_tensor(graph):
  return graph.get_tensor_by_name('pos_tensor:0')
def get_output_tensor(graph):
  policy_output = graph.get_tensor_by_name('policy_output:0')
  value_output = graph.get_tensor_by_name('value_output:0')
  return policy_output, value_output

def freeze_graph(model_path, use_trt=False, trt_max_batch_size=8,
                 trt_precision='fp32', selfplay_precision='fp32'):
    output_names = ['policy_output', 'value_output']

    n = DualNetwork(model_path)
    out_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        n.sess, n.sess.graph.as_graph_def(), output_names)

    if use_trt:
        import tensorflow.contrib.tensorrt as trt
        out_graph = trt.create_inference_graph(
            input_graph_def=out_graph,
            outputs=output_names,
            max_batch_size=trt_max_batch_size,
            max_workspace_size_bytes=1 << 29,
            precision_mode=trt_precision)

    metadata = make_model_metadata({
        'engine': 'tf',
        'use_trt': bool(use_trt),
    })

    if(selfplay_precision == 'fp32'):
         minigo_model.write_graph_def(out_graph, metadata,
                                      model_path + '.minigo')
    else:
         with tf.io.gfile.GFile(model_path  + '.pb', 'wb') as write_f:
             write_f.write(out_graph.SerializeToString())


def freeze_graph_tpu(model_path):
    """Custom freeze_graph implementation for Cloud TPU."""

    pass
#    assert model_path
#    assert FLAGS.tpu_name
#    if FLAGS.tpu_name.startswith('grpc://'):
#        tpu_grpc_url = FLAGS.tpu_name
#    else:
#        tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
#            FLAGS.tpu_name, zone=None, project=None)
#        tpu_grpc_url = tpu_cluster_resolver.get_master()
#    sess = tf.compat.v1.Session(tpu_grpc_url)
#
#    output_names = []
#    with sess.graph.as_default():
#        # Replicate the inference function for each TPU core.
#        replicated_features = []
#        feature_type = tf.bool if FLAGS.bool_features else tf.float32
#        for i in range(FLAGS.num_tpu_cores):
#            name = 'pos_tensor_%d' % i
#            features = tf.compat.v1.placeholder(
#                feature_type, [None], name=name)
#            replicated_features.append((features,))
#        outputs = contrib_tpu.replicate(
#            tpu_model_inference_fn, replicated_features)
#
#        # The replicate op assigns names like output_0_shard_0 to the output
#        # names. Give them human readable names.
#        for i, (policy_output, value_output, _) in enumerate(outputs):
#            policy_name = 'policy_output_%d' % i
#            value_name = 'value_output_%d' % i
#            output_names.extend([policy_name, value_name])
#            tf.identity(policy_output, policy_name)
#            tf.identity(value_output, value_name)
#
#        tf.compat.v1.train.Saver().restore(sess, model_path)
#
#    out_graph = tf.graph_util.convert_variables_to_constants(
#        sess, sess.graph.as_graph_def(), output_names)
#
#    metadata = make_model_metadata({
#        'engine': 'tpu',
#        'num_replicas': FLAGS.num_tpu_cores,
#    })
#
#    minigo_model.write_graph_def(out_graph, metadata, model_path + '.minigo')

def convert_pb_to_minigo(pb_path, dst_path):
    with gfile.Open(pb_path, 'rb') as f:
        model_bytes = f.read()

    metadata = make_model_metadata({
        'engine': 'tf',
        'use_trt': False,
    })

    minigo_model.write_model_bytes(model_bytes, metadata, dst_path)

def make_model_metadata(metadata):
    for f in ['conv_width', 'fc_width', 'trunk_layers', 'use_SE', 'use_SE_bias',
              'use_swish', 'input_features', 'input_layout']:
        metadata[f] = getattr(FLAGS, f)
    metadata['input_type'] = 'bool' if FLAGS.bool_features else 'float'
    metadata['board_size'] = go.N
    return metadata
