import os
import sys
import logging
import functools
import tensorflow as tf

import dual_net
import go

from multiprocessing.dummy import Pool as ThreadPool
from absl import app, flags

TF_RECORD_CONFIG = tf.compat.v1.python_io.TFRecordOptions(
    tf.compat.v1.python_io.TFRecordCompressionType.ZLIB)

flags.DEFINE_integer('batch_size', 4096,
                     'Batch size to use for data format convert.')

flags.DEFINE_string('golden_chunk_dir', None,
                    'Training example directory.')

flags.DEFINE_string('selfplay_dir', None,
                    'Selfplay example directory.')

flags.DEFINE_string('prev_layout', 'nchw',
                    'Layout of previous input features: "nhwc" or "nchw"')

flags.DEFINE_integer('shuffle_buffer_size', 2000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_boolean('shuffle_examples', True,
                     'Whether to shuffle training examples.')

FLAGS = flags.FLAGS

def make_tf_example(features, pi, value):
    """
    Args:
        features: [N, N, FEATURE_DIM] nparray of uint8
        pi: [N * N + 1] nparray of float32
        value: float
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[features.tostring()])),
        'pi': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[pi.tostring()])),
        'outcome': tf.train.Feature(
            float_list=tf.train.FloatList(
                value=[value]))}))


def write_tf_examples(filename, tf_examples, serialize=True):
    """
    Args:
        filename: Where to write tf.records
        tf_examples: An iterable of tf.Example
        serialize: whether to serialize the examples.
    """
    with tf.compat.v1.python_io.TFRecordWriter(
            filename, options=TF_RECORD_CONFIG) as writer:
        for ex in tf_examples:
            if serialize:
                writer.write(ex.SerializeToString())
            else:
                writer.write(ex)

def batch_parse_tf_example(batch_size, prev_layout,
                           layout, example_batch):
    """
    Args:
        batch_size: batch size
        layout: 'nchw' or 'nhwc'
        example_batch: a batch of tf.Example
    Returns:
        A tuple (feature_tensor, dict of output tensors)
    """
    planes = dual_net.get_features_planes()

    features = {
        'x': tf.compat.v1.FixedLenFeature([], tf.string),
        'pi': tf.compat.v1.FixedLenFeature([], tf.string),
        'outcome': tf.compat.v1.FixedLenFeature([], tf.float32),
    }
    parsed = tf.compat.v1.parse_example(example_batch, features)
    x = tf.compat.v1.decode_raw(parsed['x'], tf.uint8)

    if prev_layout == 'nhwc':
        shape = [batch_size, go.N, go.N, planes]
    else:
        shape = [batch_size, planes, go.N, go.N]
    x = tf.reshape(x, shape)

    if layout == 'nhwc':
        x = tf.transpose(x, [0, 2, 3, 1])

    pi = tf.compat.v1.decode_raw(parsed['pi'], tf.float32)
    pi = tf.reshape(pi, [batch_size, go.N * go.N + 1])
    outcome = parsed['outcome']
    return x, {'pi_tensor': pi, 'value_tensor': outcome}

def read_tf_records(batch_size, tf_records, interleave=True):
    """
    Args:
        batch_size: batch size to return
        tf_records: a list of tf_record filenames
        interleave: iwhether to interleave examples from multiple tf_records
    Returns:
        a tf dataset of batched tensors
    """

    record_list = tf.compat.v1.data.Dataset.from_tensor_slices(tf_records)

    # compression_type here must agree with write_tf_examples
    map_func = functools.partial(
        tf.compat.v1.data.TFRecordDataset,
        buffer_size=8 * 1024 * 1024,
        compression_type='ZLIB')

    if interleave:
        dataset = record_list.apply(tf.compat.v1.data.experimental.parallel_interleave(
            map_func, cycle_length=64, sloppy=True))
    else:
        dataset = record_list.flat_map(map_func)

    dataset = dataset.batch(batch_size)
    return dataset

def get_dataset(record_paths):
    dataset = read_tf_records(
                  FLAGS.batch_size,
                  record_paths,
                  interleave=False)
    dataset = dataset.map(
                  functools.partial(batch_parse_tf_example, -1,
                      FLAGS.prev_layout, FLAGS.input_layout))
    return dataset

def convert_records(dst_file):
    graph = tf.Graph()
    with graph.as_default():
        dataset = get_dataset([dst_file])
        iterator = dataset.make_one_shot_iterator()

    tf_examples = []
    with tf.compat.v1.Session(graph=graph) as sess:
        while True:
            try:
                data = sess.run(iterator.get_next())
                for i in range(0, data[0].shape[0]):
                    tf_examples.append(make_tf_example(data[0][i],
                        data[1]['pi_tensor'][i], data[1]['value_tensor'][i]))
            except tf.errors.OutOfRangeError:
                break

    write_tf_examples(dst_file, tf_examples)

def main(unused_argv):
    if(FLAGS.input_layout == FLAGS.prev_layout):
        logging.info("The src format is consistent with the dest format.")
        logging.info("Data format conversion is skipped.")
        sys.exit()

    logging.info("Converting data format from {0} to {1}".format(
        FLAGS.prev_layout, FLAGS.input_layout))

    record_pattern = os.path.join(FLAGS.selfplay_dir,
                        '*', '*', '*', '*.tfrecord.zz')
    record_list = tf.io.gfile.glob(record_pattern)


    pool = ThreadPool()

    pool.map(convert_records, record_list)

    pool.close()
    pool.join()

    logging.info("Conversion finished.")

if __name__ == '__main__':
    app.run(main)
