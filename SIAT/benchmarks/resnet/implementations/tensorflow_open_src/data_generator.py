import tensorflow as tf
import os,sys
import preprocessing
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.util import nest
#import horovod.tensorflow as hvd

class DataGenerator(object):

  def __init__(self,params):
    self._params = params

    filename_pattern = os.path.join(self._params['data_dir'], '%s-*')
    self.train_filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
    self.eval_filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
   # num_training_samples = get_num_records(self.train_filenames)
    num_training_samples = 1281167
    params['num_training_samples'] = num_training_samples
    self.num_evaluating_samples = get_num_records(self.eval_filenames)
    print ('evaluate samples:', self.num_evaluating_samples)
    params['num_evaluate_samples'] = self.num_evaluating_samples
    # ---- calculate some parameters for training ------
    global_batch_size = params['global_batch_size']
    training_steps_per_epoch = num_training_samples // global_batch_size

    # dummy image 
    with open("/cache/user-job-dir/aibench_0605_cloud/padding_new.JPEG", "rb") as image:
        self.padding_example = image.read()
    print ('--- finish reading padding new jpeg------')


    params['training_steps_per_epoch'] = training_steps_per_epoch
    params['total_training_steps'] = params['train_epochs'] * training_steps_per_epoch
    params['training_steps_between_evals'] = params['eval_interval_epochs'] * training_steps_per_epoch
    params['eval_steps_per_epoch'] = (self.num_evaluating_samples // params['global_batch_size'] )  + 1
    params['total_steps_per_eval'] = params['training_steps_between_evals'] + params['eval_steps_per_epoch']


    params['total_epochs'] = ( params['train_epochs'] // params['eval_interval_epochs'] ) * params['eval_interval_epochs']
    params['total_steps'] = ( params['train_epochs'] // params['eval_interval_epochs'] ) * params['total_steps_per_eval']
 #   params['total_eval_steps'] = ( params['train_epochs'] // params['train_and_eval_epochs'] ) * params['eval_steps']
 #   params['total_eval_epochs'] = params['train_epochs'] // params['train_and_eval_epochs'] 
    print ('total_training_steps: %d' % params['total_training_steps'])
    print ('training_steps_between_evals: %d' % params['training_steps_between_evals'])
    print ('eval_steps_per_epoch: %d' % params['eval_steps_per_epoch'])
    print ('total_steps_per_eval: %d' % params['total_steps_per_eval'])
    print ('total_epochs: %d' % params['total_epochs'])
    print ('total_steps: %d' % params['total_steps'])



  def create_train_dataset(self, batch_size, training=True, num_threads=10, increased_aug=False, shard = True):
    if self._params['use_synthetic']:
        input_shape = [224,224,3]
        input_element = nest.map_structure( lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape) )
        label_element = nest.map_structure( lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([1]) )
        element = (input_element, label_element)
        ds = tf.data.Dataset.from_tensors(element).repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size = 1)
    else:
        shuffle_buffer_size = 10000 
        num_readers = 96
        if training:
            filenames = self.train_filenames
        else:
            filenames = self.eval_filenames
      #  if hvd.size() > len(filenames):
      #      assert (hvd.size() % len(filenames)) == 0
      #      filenames = filenames * (hvd.size() / len(filenames))
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        if True:
            ######### npu ###########
            rank_size = int(os.getenv('RANK_SIZE'))
            rank_id = int(os.getenv('RANK_ID').split("-")[-1])
            device_id = int(os.getenv('DEVICE_ID')) + rank_id * 8

            ######### npu ###########
            # split the dataset into parts for each GPU
            ds = ds.shard( rank_size, device_id )

        if not training:
            ds = ds.take(self.num_evaluating_samples)  # make sure all ranks have the same amount

        if training:
            ds = ds.shuffle(1000, seed=7 * (1 + device_id ))

        ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
        counter = tf.data.Dataset.range(sys.maxsize)
        ds = tf.data.Dataset.zip((ds, counter)) 

        if training:
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size, seed=5*(1+ device_id)))
        else:
            ds = pad_dataset(ds, self._params, self.padding_example )
            ds = ds.repeat()

    
        ds = ds.map(lambda image, label: parse_record(image, training), num_parallel_calls=14)
        #ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE) # split device here
        #ds = ds.map(lambda image, label: parse_record1(image, label), num_parallel_calls=14)
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds
  
  def make_iterator_initialize(self,training=True):
    ds = self.create_train_dataset(self._params['batch_size'], training, shard=training)
    self.multi_device_iterator = ds.make_initializable_iterator()
    return self.multi_device_iterator


#    self.multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(ds, self._params['gpu_devices'], source_device='/cpu:0')
# #   self.multi_device_iterator = self.strategy.make_dataset_iterator(dataset=ds)
#    iterator_init_op = self.multi_device_iterator.initializer
#
##    self.multi_device_iterator = tf.data.make_initializable_iterator(dataset=ds)
##    iterator_init_op = self.multi_device_iterator.initializer
#    return iterator_init_op

  def get_next(self):
    return self.multi_device_iterator.get_next()
  
  def get_staged_next(self):
    next_tensor = self.multi_device_iterator.get_next()

    tensor_dtypes = [ (image.dtype, label.dtype) for (image,label) in next_tensor ]
    tensor_shapes = [ (image.shape, label.shape) for (image,label) in next_tensor ]
    
    with tf.device('/cpu:0'):
      stage_area = data_flow_ops.StagingArea( dtypes=tensor_dtypes, shapes=tensor_shapes )
      put_op = stage_area.put(next_tensor) 
      get_tensors = stage_area.get()

    return put_op, get_tensors
    



def get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count
    nfile = len(filenames)
    total = 0
    for i in filenames:
        num = count_records(i)
        total = total + num
    return total

def pad_dataset(dataset, config, padding_example):
    import math
    num_dataset_per_shard = int(
        config['eval_steps_per_epoch'] * config['batch_size'])  #每卡测试样本数

    print ('--- num_dataset_per_shard:', num_dataset_per_shard)
    padded_example = _convert_to_example(
        padding_example, -100).SerializeToString()  #pad的样本 label是-100
    print ('--- padded example:', padded_example)
    padded_dataset = tf.data.Dataset.from_tensors(
        tf.constant(padded_example, dtype=tf.string))
    print ('--- padded dataset:', padded_dataset)

    padded_dataset = padded_dataset.repeat(num_dataset_per_shard)
    print ('--- padded dataset repeat:', padded_dataset)
    counter = tf.data.Dataset.range(sys.maxsize)
    padded_dataset = tf.data.Dataset.zip((padded_dataset, counter)) 
    print ('--- padded dataset dataset:', dataset)
    dataset = dataset.concatenate(padded_dataset).take(num_dataset_per_shard)
    print ('--- padded dataset repeat concat:', dataset)
    return dataset

def _convert_to_example(image_buffer, label):
    """Build an Example proto for an example.

    Args:
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network

    Returns:
        Example proto
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/class/label': _int64_feature(label),
                'image/encoded': _bytes_feature(image_buffer)
            }))
    return example

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _parse_example_proto(example_serialized):
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox

def parse_record(raw_record,is_training):
  image_buffer, label, bbox = _parse_example_proto(raw_record)
  # for 1980 only
  config={'min_object_covered': 0.1, 'aspect_ratio_range': [3. / 4., 4. / 3.], 'area_range': [0.45, 1.0], 'max_attempts': 100}
  if is_training:
    image_hw = 160
  else:
    image_hw = 224
  image = preprocessing.parse_and_preprocess_image_record(
    config, image_buffer, height=image_hw, width=image_hw,
    brightness=0.3, contrast=0.6, saturation=0.6, hue=0.13,
    distort=is_training, nsummary=10, increased_aug=False, random_search_aug=False)
  return image, label

def parse_record1(image, label):
  image = preprocessing.split_device(image)
  return image, label




