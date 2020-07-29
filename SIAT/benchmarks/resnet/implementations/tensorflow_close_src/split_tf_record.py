import tensorflow as tf
import os

data_dir = '/data/Datasets/imagenet_TF'
new_dir = '/development/h00452838/resnet50_training_padding_eval/data_loader/resnet50/split_eval_data'

filename_pattern = os.path.join(data_dir, '%s-*')
eval_filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

part_num = 0
split_size = 1024


def split_tfrecord(tfrecord_path, split_size):
    global part_num
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        while True:
            try:
                records = sess.run(batch)
                # part_path = tfrecord_path + '.{:03d}'.format(part_num)
                part_path = new_dir+'/validation'+'-{:05d}'.format(part_num)+'-of-01024'
                print(part_path)
                with tf.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break

for i in eval_filenames:
    split_tfrecord(i, split_size)


