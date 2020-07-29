"""Convert checkpoint file to new format."""

from absl import app
from absl import flags
import numpy as np
import tensorflow.google.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("old", None, "old checkpoint file")
flags.DEFINE_string("new", None, "new checkpoint file")


def get_new_shape(n, s):
  if n.find("attention/output") >= 0:
    return [16, 64, 1024] if len(s) == 2 else [1024]
  if n.find("query") >= 0 or n.find("key") >= 0 or n.find("value") >= 0:
    return [1024, 16, 64] if len(s) == 2 else [16, 64]
  return s


def main(unused_argv):
  reader = tf.train.NewCheckpointReader(FLAGS.old)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()

  tf.reset_default_graph()
  prefix = "bert/encoder/layer_"
  with tf.Session() as sess:
    for n in shapes:
      if n.startswith(prefix):
        if n.startswith(prefix + "0") and n.find("LAMB") < 0:
          if n.find("kernel") < 0:
            suffix = n[len(prefix + "0"):]
            new_name = "bert/encoder/layer_common" + suffix
            for m in ["", "/LAMB", "/LAMB_1"]:
              new_shape = [24] + get_new_shape(n, shapes[n])
              v = tf.get_variable(new_name + m, new_shape, dtypes[n])
              tf.keras.backend.set_value(
                  v,
                  np.array([
                      reader.get_tensor(prefix + str(i) + suffix + m)
                      for i in range(24)
                  ]).reshape(new_shape))
          else:
            suffix = n[len(prefix + "0"):n.rfind("/")] + "/layer_"
            keyword = n[n.rfind("/"):]
            for m in ["", "/LAMB", "/LAMB_1"]:
              for i in range(24):
                old_name = prefix + str(i) + n[len(prefix) + 1:] + m
                new_name = "bert/encoder/layer_common" + suffix + str(
                    i) + keyword + m
                new_shape = get_new_shape(n, shapes[n])
                tf.keras.backend.set_value(
                    tf.get_variable(new_name, new_shape, dtypes[n]),
                    np.array(reader.get_tensor(old_name)).reshape(new_shape))
      else:
        new_shape = get_new_shape(n, shapes[n])
        tf.keras.backend.set_value(
            tf.get_variable(n, new_shape, dtypes[n]),
            np.array(reader.get_tensor(n)).reshape(new_shape))
    tf.train.Saver().save(sess, FLAGS.new)


if __name__ == "__main__":
  app.run(main)
