import tensorflow as tf
#import horovod.tensorflow as hvd
from tensorflow.contrib.image.python.ops import distort_image_ops
import math
#from .data_aug_search import random_aug_search



def deserialize_image_record(record): 
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        bbox = tf.stack([obj['image/object/bbox/%s' % x].values
                         for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
        text = obj['image/class/text']
        return imgdata, label, bbox, text

def decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')


def crop_and_resize_image(config, image, height, width, 
                          distort=False, nsummary=10):
    with tf.name_scope('crop_and_resize'):
        # Evaluation is done on a center-crop of this ratio
        eval_crop_ratio = 1.0
        if distort:
            initial_shape = [int(round(height / eval_crop_ratio)),
                             int(round(width / eval_crop_ratio)),
                             3]
            jpeg_shape = tf.image.extract_jpeg_shape( image )

            bbox_begin, bbox_size, bbox = \
                tf.image.sample_distorted_bounding_box(
                    initial_shape,
                    bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
                    # tf.zeros(shape=[1,0,4]), # No bounding boxes
                    min_object_covered=config['min_object_covered'],
                    aspect_ratio_range=config['aspect_ratio_range'],
                    area_range=config['area_range'],
                    max_attempts=config['max_attempts'],
                 #   seed=11 ,  # Need to set for deterministic results
                    use_image_if_no_bounding_boxes=True)
            bbox = bbox[0, 0]  # Remove batch, box_idx dims


            y_min = tf.cast( bbox[0] * (tf.cast( jpeg_shape[0], tf.float32) ), tf.int32)
            x_min = tf.cast( bbox[1] * (tf.cast(jpeg_shape[1], tf.float32) ), tf.int32) 
            y_max = tf.cast( bbox[2] * (tf.cast(jpeg_shape[0], tf.float32) ), tf.int32)
            x_max = tf.cast( bbox[3] * (tf.cast(jpeg_shape[1], tf.float32) ), tf.int32)

            crop_height = y_max - y_min
            crop_width = x_max - x_min
            crop_window = tf.stack( [y_min, x_min, crop_height, crop_width] )
            image = tf.image.decode_and_crop_jpeg( image, crop_window, channels=3 )
            image = tf.image.resize_images( image, [height, width] )
            
            
        else:
            # Central crop

            image = decode_jpeg(image, channels=3)


            image = aspect_preserve_resize( image, 256 )

            image = central_crop(image, height, width)


#            ratio_y = ratio_x = eval_crop_ratio
#            bbox = tf.constant([0.5 * (1 - ratio_y), 0.5 * (1 - ratio_x),
#                                0.5 * (1 + ratio_y), 0.5 * (1 + ratio_x)])
#            image = tf.image.crop_and_resize(
#               image[None, :, :, :], bbox[None, :], [0], [height, width])[0]
        
        return image

def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
  
    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])



def aspect_preserve_resize(image, resize_min):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least( height, width, resize_min )
    image = tf.image.resize_images( image, [new_height, new_width] )
    return image


def _smallest_size_at_least(height, width, resize_min):

    resize_min = tf.cast(resize_min, tf.float32)
  
    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
  
    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim
  
    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    return new_height, new_width



def parse_and_preprocess_image_record(config, record, height, width,
                                      brightness, contrast, saturation, hue,
                                      distort, nsummary=10, increased_aug=False, random_search_aug=False):
    #imgdata, label, bbox, text = deserialize_image_record(record)
    #label -= 1  # Change to 0-based (don't use background class)
    with tf.name_scope('preprocess_train'):
            image = crop_and_resize_image(config, record, height, width, distort)
            if distort:
                image = tf.image.random_flip_left_right(image)
                if increased_aug:
                    image = tf.image.random_brightness(image, max_delta=brightness)  
                    image = distort_image_ops.random_hsv_in_yiq(image, 
                                                                lower_saturation=saturation, 
                                                                upper_saturation=2.0 - saturation, 
                                                                max_delta_hue=hue * math.pi)
                    image = tf.image.random_contrast(image, lower=contrast, upper=2.0 - contrast)
                    tf.summary.image('distorted_color_image', tf.expand_dims(image, 0))
                    
    image = tf.clip_by_value(image, 0., 255.)
    image = normalize(image)
    image = tf.cast(image, tf.float16)

    return image
def normalize(inputs):
   #  imagenet_mean = [121.0, 115.0, 100.0]             #np.array([121, 115, 100], dtype=np.float32)
   #  imagenet_std =  [70.0, 68.0, 71.0]                #np.array([70, 68, 71], dtype=np.float32)
     imagenet_mean = [123.68, 116.78, 103.94]             #np.array([121, 115, 100], dtype=np.float32)
     imagenet_std =  [1.0, 1.0, 1.0]                #np.array([70, 68, 71], dtype=np.float32)
     imagenet_mean = tf.expand_dims(tf.expand_dims(imagenet_mean, 0), 0)
     imagenet_std = tf.expand_dims(tf.expand_dims(imagenet_std, 0), 0)
     inputs = inputs - imagenet_mean          #tf.subtract(inputs, imagenet_mean)
     inputs = inputs * (1.0 / imagenet_std)
     #inputs = tf.multiply(inputs, 1. / imagenet_std)

     return inputs

def split_device(image):
    image = tf.clip_by_value(image, 0., 255.)
    image = normalize(image)
    image = tf.cast(image, tf.float16)
    return image
