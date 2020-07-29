import tensorflow as tf
import resnet
import learning_rate
#import horovod.tensorflow as hvd
from npu_bridge.estimator.npu.npu_optimizer import  NPUDistributedOptimizer
from npu_bridge.estimator import npu_ops
#import optimizer

class Model(object):
  def __init__(self, params):
    self.params = params

  def model_func(self, images, labels, is_training=True, train_steps=None): 
    model_inference_func = self.get_model_func()
    with tf.name_scope('resnet') as name_scope:
        with tf.device('/gpu:0'):
          labels = tf.reshape( labels, (-1,) ) 
          image = tf.cast( images, self.params['dtype'] )

          if self.params['data_format'] == 'channels_first':
            image = tf.transpose(image, [0,3,1,2])

          logits = model_inference_func( image, self.params['data_format'], training=is_training, 
                                         conv_initializer=tf.variance_scaling_initializer(seed=1),
                                         bn_init_mode='conv_bn_init', bn_gamma_initial_value=1.0  )
          
          logits = tf.cast(logits, tf.float32)
          one_hot_labels = tf.one_hot(labels, self.params['num_classes'])
          base_loss = tf.losses.softmax_cross_entropy( one_hot_labels, logits=logits, label_smoothing=0.1 )
    
          predicted_label = tf.math.argmax(logits,1, output_type=tf.int32)
        
          # Eval branch
          if not is_training:
              return tf.no_op(name='eval_op'), predicted_label, base_loss, base_loss, train_steps, labels
  
          def exclude_batch_norm(name):
            return 'BatchNorm' not in name and 'batchnorm' not in name and 'batch_norm' not in name and 'Batch_Norm' not in name

          if self.params['use_lars']:
              total_loss = base_loss
          else:
              l2_loss = self.params['weight_decay'] * tf.add_n( 
                      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                       if exclude_batch_norm(v.name)])
              total_loss = base_loss + l2_loss
          
          lr = learning_rate.get_lr(self.params, train_steps)    
          opt = tf.train.MomentumOptimizer( lr, self.params['momentum'] )
          opt = NPUDistributedOptimizer(opt)
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []

          with tf.control_dependencies(update_ops):

            gate_gradients = (tf.train.Optimizer.GATE_NONE)
            scaled_grads = opt.compute_gradients( total_loss * 1024.0 )
            unscaled_grads = [ (g/1024.0, v) for g,v in scaled_grads ]

            if self.params['use_lars']:
                g_list_bn_bias = []
                var_list_bn_bias = []
                g_list_else = []
                var_list_else = []
                g_list_else_lars = []
                grad_var_list=[]
                for g,var in unscaled_grads:
                    if 'BatchNorm' not in var.name and 'bias' not in var.name:
                        g_list_else.append(g)
                        var_list_else.append(var)

                        g_new = npu_ops.LARSV2( input_weight=var,
                                      input_grad = g,
                                      weight_decay = self.params['weight_decay'],
                                      learning_rate = 1.0, use_clip=False )
                        g_list_else_lars.append(g_new)
                    else:
                        g_list_bn_bias.append(g)
                        var_list_bn_bias.append(var)

                g_list_lars = g_list_bn_bias + g_list_else_lars
                var_list = var_list_bn_bias + var_list_else

                for (g, var) in zip(g_list_lars,var_list):
                    g_and_v = ( g, var )
                    grad_var_list.append( g_and_v )

                train_op = opt.apply_gradients(grad_var_list)
            else:
                train_op = opt.apply_gradients(unscaled_grads)

    return train_op, predicted_label, base_loss, lr, train_steps, labels

  def get_model_func(self):
      model_name = self.params['model_name']
      if model_name.startswith('resnet'):
          nlayer = int(model_name[len('resnet'):])
          return lambda images, *args, **kwargs: \
              resnet.inference_resnet_v1(self.params,images, nlayer, *args, **kwargs)
      else:
          raise ValueError("Invalid model type: %s" % model_name)

