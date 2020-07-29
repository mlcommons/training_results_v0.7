# Lint as: python3
"""Script to train BERT model on SQuAD task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import json
import os
from absl import app
from absl import flags
from absl import logging
from flax import nn
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import tensorflow.compat.v2 as tf
from REDACTED import gfile
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import utils
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax.bert_models import SquadModel as jax_squad
from REDACTED.tensorflow_models.official.nlp.bert import configs
from REDACTED.tensorflow_models.official.nlp.bert import input_pipeline
from REDACTED.tensorflow_models.official.nlp.bert import squad_evaluate_v1_1
from REDACTED.tensorflow_models.official.nlp.bert import tokenization
from REDACTED.tensorflow_models.official.nlp.data import squad_lib

FRACTION_WARMUP_STEPS = 0.1
BERT_DIR = '/REDACTED/ym-d/home/hongkuny/public/pretrained_models/keras_bert/uncased_L-24_H-1024_A-16'
RawResult = collections.namedtuple('RawResult',
                                   ['unique_id', 'start_logits', 'end_logits'])

flags.DEFINE_string(
    'train_data_path',
    '/REDACTED/tp-d/home/tensorflow-tpus/bert/data/squad_train.tf_record',
    'Training data path with train tfrecords.')
flags.DEFINE_string(
    'input_meta_data_path',
    '/REDACTED/tp-d/home/tensorflow-tpus/bert/data/squad_meta_data',
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_string('predict_file',
                    '/REDACTED/od-d/home/lumiere/public/data/squad/dev-v1.1.json',
                    'Prediction data path with train tfrecords.')
flags.DEFINE_string('vocab_file',
                    BERT_DIR + '/vocab.txt',
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_string(
    'bert_config_file',
    BERT_DIR + '/bert_config.json',
    'Config file for BERT model.')
flags.DEFINE_string(
    'init_checkpoint',
    BERT_DIR + '/bert_model.ckpt',
    'Checkpoint of pretained model.')
flags.DEFINE_integer('train_batch_size', 4, 'Total batch size for training.')
flags.DEFINE_integer('num_train_epochs', 2, 'Total epochs for training.')
flags.DEFINE_integer('steps_per_loop', 1, 'Total number of steps per loop.')
flags.DEFINE_integer('train_steps', None, 'Total number of steps to train.')
flags.DEFINE_integer('predict_batch_size', 4, 'Total batch size for predict.')
flags.DEFINE_bool('save_checkpoint', True,
                  'Whether to save current training model checkpoints.')
flags.DEFINE_bool('load_checkpoint', False,
                  'Whether to restore from existing model checkpoints.')
flags.DEFINE_integer('checkpoint_freq', 1000,
                     'Number of steps after which we save model checkpoint.')
flags.DEFINE_integer('log_train_metrics_steps', 100,
                     'Number of steps after which we print train metrics.')
flags.DEFINE_boolean('use_real_input', True,
                     'Use real input or synthetic input.')
flags.DEFINE_boolean('load_tf_weights', True,
                     'Load tensorflow pretrained weights into JAX model.')
flags.DEFINE_boolean('do_lower_case', True,
                     'Whether to lower case the input text.')
flags.DEFINE_boolean('use_eval_sharding', False,\
                     'Whether to use sharding for eval.')
flags.DEFINE_integer(
    'n_best_size', 20,
    'Totel number of n-best predictions to generate in the '
    'nbest_predictions.json output file.')
flags.DEFINE_integer(
    'max_answer_length', 30,
    'The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.')
flags.DEFINE_float(
    'null_score_diff_threshold', 0.0,
    'If null_score - best_non_null is greater than the threshold, '
    'predict null. This is only used for SQuAD v2.')
flags.DEFINE_bool(
    'verbose_logging', False,
    'If true, all of the warnings related to data processing will be printed. '
    'A number of warnings are expected for a normal SQuAD evaluation.')
flags.DEFINE_enum(
    'mode', 'train_and_predict',
    ['train_and_predict', 'train', 'predict'],
    'One of {"train_and_predict", "train", "predict"}. '
    '`train_and_predict`: both train and predict to a json file. '
    '`train`: only trains the model. '
    '`predict`: predict answers from the squad json file. ')

flags.DEFINE_string(
    'model_dir',
    default=None,
    help=('The directory where the model and summaries are stored.'))
flags.DEFINE_float('learning_rate', 1e-4, help='Base learning rate.')

FLAGS = flags.FLAGS


def create_squad_dataset(file_path,
                         seq_length,
                         batch_size,
                         host_count,
                         host_id,
                         is_training=True):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
  }
  if is_training:
    name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)
  else:
    name_to_features['unique_ids'] = tf.io.FixedLenFeature([], tf.int64)

  dataset = input_pipeline.single_file_dataset(file_path, name_to_features)

  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  dataset = dataset.shard(host_count, host_id)

  def _select_data_from_record(record):
    """Dispatches record to features and labels."""
    x, y = {}, {}
    for name, tensor in record.items():
      if name in ('start_positions', 'end_positions'):
        y[name] = tensor
      elif name == 'input_ids':
        x['input_word_ids'] = tensor
      elif name == 'segment_ids':
        x['input_type_ids'] = tensor
      else:
        x[name] = tensor
    return (x, y)

  dataset = dataset.map(_select_data_from_record)

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1024)

  return dataset


def cross_entropy_loss(logits, labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size


def get_squad_loss(labels, model_outputs, loss_factor=1.0):
  """Calculate loss for SQuAD task.

  Args:
    labels: one-hot label of (start_positions, end_positions)
    model_outputs: output of the squad model as (start_logits, end_logits)
    loss_factor: factor applied to loss calculation

  Returns:
    total_loss: loss calculated from start and end predictions
  """
  start_positions, end_positions = labels
  start_logits, end_logits = model_outputs

  start_loss = cross_entropy_loss(start_logits, start_positions)
  end_loss = cross_entropy_loss(end_logits, end_positions)

  total_loss = (start_loss + end_loss) / 2
  total_loss *= loss_factor

  return total_loss


def get_squad_model():
  """Get SQuAD model with pretrained weights loaded.

  Returns:
    jax_squad_model: squad model with TF pretrained weights loaded to its
    transformer encoder part
  """
  # Get pretrained TF model configuration and variables from checkpoint
  if FLAGS.load_tf_weights:
    tf_config, tf_vars, _ = utils.get_tf_model_variables(
        FLAGS.bert_config_file, FLAGS.init_checkpoint)
  else:
    tf_config = configs.BertConfig.from_json_file(
        FLAGS.bert_config_file).__dict__

  # Generate JAX model using same model configuration as TF model
  rng = random.PRNGKey(0)
  sequence_length = 20
  host_batch_size = int(FLAGS.train_batch_size // jax.host_count())
  device_batch_size = int(host_batch_size // jax.local_device_count())
  model_kwargs = utils.convert_tf_config_to_jax_bert(tf_config)
  # Not defined by jax bert.
  with nn.stochastic(rng):
    model_def = jax_squad.partial(**model_kwargs)
    _, jax_squad_model = model_def.create_by_shape(
        rng, [((3, device_batch_size, sequence_length), jnp.int32)])

  # Update transformer encoder parameters with TF model pretrained weights
  if FLAGS.load_tf_weights:
    jax_transformer_vars = utils.convert_tf_param_dict_to_jax(tf_vars)
    jax_squad_model.params.update({'transformer_encoder': jax_transformer_vars})
  else:
    encoder_vars = jax_squad_model.params['transformer_encoder']
    encoder_vars['self_attention_mask'] = 0.0
    jax_squad_model.params.update({'transformer_encoder': encoder_vars})

  return jax_squad_model


def create_optimizer(model, learning_rate=1e-4):
  """Create optimizer used for training model.

  MultiOpt is used to apply Adam Optimizer with weight decay to all parameters
  except layer_norm and bias and Adam Optimizer without weight decay for
  layer_norm and bias params.

  Args:
    model: JAX model to add optimizer to
    learning_rate: base learning rate used for initializing optimizer

  Returns:
    optimizer: model with Adam Optimizer to be used for training
  """
  weight_decay_def = optim.Adam(
      learning_rate=learning_rate, eps=1e-6, weight_decay=0.01)
  no_decay_def = optim.Adam(
      learning_rate=learning_rate, eps=1e-6, weight_decay=0.0)

  def filter_weight_decay(key, _):
    return 'layer_norm' not in key and 'bias' not in key
  def filter_other(key, _):
    return 'layer_norm' in key or 'bias' in key

  weight_decay_traversal = optim.ModelParamTraversal(filter_weight_decay)
  no_decay_traversal = optim.ModelParamTraversal(filter_other)
  optimizer_def = optim.MultiOptimizer(
      (weight_decay_traversal, weight_decay_def),
      (no_decay_traversal, no_decay_def))

  optimizer = optimizer_def.create(model)
  optimizer = optimizer.replicate()
  del model
  return optimizer


def create_learning_rate_scheduler(
    base_learning_rate=0.5,
    warmup_steps=1000,
    total_training_steps=100000):
  """Create scheduler for learning rate."""

  def step_fn(step):
    """Step to learning rate function."""
    lr = base_learning_rate
    return lax.cond(step < warmup_steps,
                    lr, lambda x: x * (step / warmup_steps),
                    lr, lambda x: x * (1.0 - step / total_training_steps))

  return step_fn


@jax.jit
def local_barrier_helper():
  return jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count()))


def local_barrier():
  local_barrier_helper().block_until_ready()


@jax.jit
def global_barrier_helper():
  d = jax.random.normal(jax.random.PRNGKey(0), [jax.local_device_count()])
  val = jax.pmap(lambda x: jax.lax.psum(x, axis_name='i'), axis_name='i')(d)
  return val


def global_barrier():
  val = global_barrier_helper()
  local_barrier()
  return val


def train_step(optimizer, inputs, labels, learning_rate_fn, dropout_rng=None):
  """A single training step.

  Args:
    optimizer: optimizer used for training
    inputs: inputs to the model [word_ids, mask, type_ids]
    labels: target output [start_positions, end_positions]
    learning_rate_fn: function for tuning learning rate
    dropout_rng: random seed used for dropout

  Returns:
    new_optimizer: updated model optimizer after training step
    loss: sparse categorical crossentropy
    new_dropout_rng: new random seed to be used for next step
  """
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True)
    loss = get_squad_loss(labels, logits)
    return loss, logits

  step = optimizer.state[0].step
  lr = learning_rate_fn(step)
  new_optimizer, loss, _ = optimizer.optimize(
      loss_fn, learning_rate=lr)

  return new_optimizer, loss, new_dropout_rng


def train_squad(optimizer, input_meta_data, start_step):
  """Run bert squad training.

  Args:
    optimizer: BERT model with squad layer
    input_meta_data: dictionary with input meta data
    start_step: start step of training loop, 0 if no checkpoint was loaded

  Returns:
    optimizer: trained model
  """

  # Get input dataset and configuration
  epochs = FLAGS.num_train_epochs
  num_train_examples = input_meta_data['train_data_size'] // jax.host_count()
  max_seq_length = input_meta_data['max_seq_length']
  host_batch_size = int(FLAGS.train_batch_size // jax.host_count())
  steps_per_epoch = int(num_train_examples // host_batch_size)
  device_count = jax.local_device_count()
  rng = random.PRNGKey(0)

  if FLAGS.use_real_input:
    dataset = create_squad_dataset(
        FLAGS.train_data_path,
        max_seq_length,
        host_batch_size,
        jax.host_count(),
        jax.host_id(),
        is_training=True)
    train_iterator = iter(dataset)
  else:
    vocab_size = 10
    type_size = 5
    input_shape = (host_batch_size, max_seq_length)
    word_id_data = random.randint(rng, input_shape, 0, vocab_size)
    mask_data = random.randint(rng, input_shape, 0, 2)
    type_id_data = random.randint(rng, input_shape, 0, type_size)

    inputs = [word_id_data, type_id_data, mask_data]
    labels = [random.randint(rng, (host_batch_size,), 0, 1)] * 2
    inputs = common_utils.shard(inputs)
    labels = common_utils.shard(labels)

  # train step
  total_training_steps = steps_per_epoch * epochs if FLAGS.train_steps is None else FLAGS.train_steps
  warmup_steps = int(epochs * num_train_examples * FRACTION_WARMUP_STEPS //
                     host_batch_size)
  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=FLAGS.learning_rate,
      warmup_steps=warmup_steps,
      total_training_steps=total_training_steps)
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')

  dropout_rngs = random.split(rng, device_count)
  for step in range(start_step, total_training_steps):
    if FLAGS.use_real_input:
      inputs, labels = next(train_iterator)
      inputs = [
          inputs['input_word_ids'].numpy(), inputs['input_mask'].numpy(),
          inputs['input_type_ids'].numpy()
      ]
      labels = [
          labels['start_positions'].numpy(), labels['end_positions'].numpy()
      ]
      inputs = common_utils.shard(inputs)
      labels = common_utils.shard(labels)

    optimizer, loss, dropout_rngs = p_train_step(optimizer, inputs, labels,
                                                 dropout_rng=dropout_rngs)

    if (step + 1) % FLAGS.log_train_metrics_steps == 0:
      logging.info('(Step %s / %s) Loss: %s', step + 1, total_training_steps,
                   jnp.mean(loss))

    if (step + 1) % FLAGS.checkpoint_freq == 0 and FLAGS.save_checkpoint:
      if jax.host_id() == 0:
        checkpoints.save_checkpoint(FLAGS.model_dir, optimizer, step,
                                    prefix='checkpoint', keep=1)
  return optimizer


def get_raw_results(predictions):
  """Get raw results from model predictions."""
  for unique_ids, start_logits, end_logits in zip(*predictions):
    yield RawResult(
        unique_id=unique_ids,
        start_logits=start_logits.tolist(),
        end_logits=end_logits.tolist())


def get_raw_results_sharded(predictions):
  """Get raw results from model predictions in sharded form."""
  for unique_ids, start_logits, end_logits in zip(*predictions):
    for values in zip(unique_ids, start_logits, end_logits):
      yield RawResult(
          unique_id=values[0],
          start_logits=values[1].tolist(),
          end_logits=values[2].tolist())


def predict_step(model, inputs):
  """A single predict step."""
  logits = model(inputs, train=False)
  return logits


def predict_squad_customized(optimizer, input_meta_data,
                             predict_tfrecord_path, num_steps):
  """Make predictions using BERT squad model.

  Args:
    optimizer: BERT model with squad layer
    input_meta_data: dictionary with input meta data
    predict_tfrecord_path: file path for predict tf record
    num_steps: number of eval steps

  Returns:
    all_results: all raw results from the model
  """
  dataset_fn = create_squad_dataset(
      predict_tfrecord_path,
      input_meta_data['max_seq_length'],
      FLAGS.predict_batch_size,
      host_count=1,  # don't use split dataset for eval
      host_id=0,  # don't use split dataset for eval
      is_training=False)
  predict_iterator = iter(dataset_fn)

  if FLAGS.use_eval_sharding:
    p_predict_step = jax.pmap(
        functools.partial(predict_step), axis_name='batch')

  all_results = []
  for _ in range(num_steps):
    inputs, _ = next(predict_iterator)
    unique_ids = inputs.pop('unique_ids').numpy()
    inputs = [
        inputs['input_word_ids'].numpy(), inputs['input_mask'].numpy(),
        inputs['input_type_ids'].numpy()
    ]

    # If using sharding for eval, shard inputs and use pmapped eval function
    if FLAGS.use_eval_sharding:
      inputs = common_utils.shard(inputs)
      unique_ids = common_utils.shard(unique_ids)
      predictions = p_predict_step(optimizer.target, inputs)
      predictions = (unique_ids,) + predictions
      for result in get_raw_results_sharded(predictions):
        all_results.append(result)
    else:
      predictions = predict_step(optimizer.target, inputs)
      predictions = (unique_ids,) + predictions
      for result in get_raw_results(predictions):
        all_results.append(result)

    if len(all_results) % 100 == 0:
      logging.info('Made predictions for %d records.', len(all_results))

  return all_results


def get_f1_score(dataset_file_path, predictions_file_path):
  """Calculate exact match and F1 score on prediction output."""
  with gfile.Open(dataset_file_path) as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
  with gfile.Open(predictions_file_path) as prediction_file:
    predictions = json.load(prediction_file)

  logging.info('Eval F1 score')
  logging.info(json.dumps(squad_evaluate_v1_1.evaluate(dataset, predictions)))


def predict_squad(optimizer, input_meta_data):
  """Run bert squad predict."""
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  doc_stride = input_meta_data['doc_stride']
  max_query_length = input_meta_data['max_query_length']
  version = input_meta_data.get('version_2_with_negative', False)
  eval_examples = squad_lib.read_squad_examples(
      input_file=FLAGS.predict_file,
      is_training=False,
      version_2_with_negative=version)
  eval_writer = squad_lib.FeatureWriter(
      filename=os.path.join(FLAGS.model_dir, 'eval.tf_record'),
      is_training=False)
  eval_features = []

  def _append_feature(feature, is_padding):
    if not is_padding:
      eval_features.append(feature)
    eval_writer.process_feature(feature)

  kwargs = dict(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=input_meta_data['max_seq_length'],
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=False,
      output_fn=_append_feature,
      batch_size=FLAGS.predict_batch_size)

  dataset_size = squad_lib.convert_examples_to_features(**kwargs)
  eval_writer.close()

  num_steps = int(dataset_size / FLAGS.predict_batch_size)
  all_results = predict_squad_customized(optimizer, input_meta_data,
                                         eval_writer.filename, num_steps)

  logging.info('Saving predictions to: %s', FLAGS.model_dir)
  output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
  output_nbest_file = os.path.join(FLAGS.model_dir, 'nbest_predictions.json')
  output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')

  squad_lib.write_predictions(
      eval_examples,
      eval_features,
      all_results,
      FLAGS.n_best_size,
      FLAGS.max_answer_length,
      FLAGS.do_lower_case,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      version_2_with_negative=version,
      null_score_diff_threshold=FLAGS.null_score_diff_threshold,
      verbose=FLAGS.verbose_logging)

  get_f1_score(FLAGS.predict_file, output_prediction_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not gfile.IsDirectory(FLAGS.model_dir):
    gfile.MakeDirs(os.path.dirname(FLAGS.model_dir))

  logging.info('Number of recognized devices: %d', jax.local_device_count())
  logging.info('Import pretrained weights: %s', FLAGS.load_tf_weights)
  jax_squad_model = get_squad_model()

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  step = 0
  optimizer = create_optimizer(jax_squad_model, FLAGS.learning_rate)
  if FLAGS.load_checkpoint:
    optimizer = checkpoints.restore_checkpoint(FLAGS.model_dir, optimizer)
    step = optimizer.state[0].step[0]

  if FLAGS.mode in ('train', 'train_and_predict'):
    optimizer = train_squad(optimizer, input_meta_data, step)

  if FLAGS.mode in ('predict', 'train_and_predict'):
    if not FLAGS.use_eval_sharding:
      optimizer = optimizer.unreplicate()
    if jax.host_id() == 0:
      predict_squad(optimizer, input_meta_data)

  global_barrier()


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
