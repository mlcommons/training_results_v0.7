import tensorflow as tf
# from mlperf_compliance import mlperf_log
# from mlperf_compliance import resnet_log_helper
# from configs.res50.res50_config import res50_config

_BATCH_NORM_EPSILON = 1e-4
_BATCH_NORM_DECAY = 0.9

ML_PERF_LOG=False

class LayerBuilder(object):
    def __init__(self, activation=None, data_format='channels_last',
                 training=False, use_batch_norm=False, batch_norm_config=None,
                 conv_initializer=None, bn_init_mode='adv_bn_init', bn_gamma_initial_value=1.0 ):
        self.activation = activation
        self.data_format = data_format
        self.training = training
        self.use_batch_norm = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self.conv_initializer = conv_initializer
        self.bn_init_mode = bn_init_mode
        self.bn_gamma_initial_value = bn_gamma_initial_value
        if self.batch_norm_config is None:
            self.batch_norm_config = {
                'decay': _BATCH_NORM_DECAY,
                'epsilon': _BATCH_NORM_EPSILON,
                'scale': True,
                'zero_debias_moving_mean': False,
            }

    def _conv2d(self, inputs, activation, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=not self.use_batch_norm,
            kernel_initializer=self.conv_initializer,
            activation=None if self.use_batch_norm else activation,
            *args, **kwargs)
        if self.use_batch_norm:
            param_initializers = {
                'moving_mean': tf.zeros_initializer(),
                'moving_variance': tf.ones_initializer(),
                'beta': tf.zeros_initializer(),
            }
            if self.bn_init_mode == 'adv_bn_init':
                param_initializers['gamma'] = tf.ones_initializer()
            elif self.bn_init_mode == 'conv_bn_init':
                param_initializers['gamma'] = tf.constant_initializer(self.bn_gamma_initial_value)
            else:
                raise ValueError("--bn_init_mode must be 'conv_bn_init' or 'adv_bn_init' ")

            x = self.batch_norm(x)
            x = activation(x) if activation is not None else x
        return x

    def conv2d_linear_last_bn(self, inputs, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=False,
            kernel_initializer=self.conv_initializer,
            activation=None, *args, **kwargs)
        param_initializers = {
            'moving_mean': tf.zeros_initializer(),
            'moving_variance': tf.ones_initializer(),
            'beta': tf.zeros_initializer(),
        }
        if self.bn_init_mode == 'adv_bn_init':
            param_initializers['gamma'] = tf.zeros_initializer()
        elif self.bn_init_mode == 'conv_bn_init':
            param_initializers['gamma'] = tf.constant_initializer(self.bn_gamma_initial_value)
        else:
            raise ValueError("--bn_init_mode must be 'conv_bn_init' or 'adv_bn_init' ")    

        x = self.batch_norm(x, param_initializers=param_initializers)
        return x

    def conv2d_linear(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, None, *args, **kwargs)

    def conv2d(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, self.activation, *args, **kwargs)

    def pad2d(self, inputs, begin, end=None):
        if end is None:
            end = begin
        try:
            _ = begin[1]
        except TypeError:
            begin = [begin, begin]
        try:
            _ = end[1]
        except TypeError:
            end = [end, end]
        if self.data_format == 'channels_last':
            padding = [[0, 0], [begin[0], end[0]], [begin[1], end[1]], [0, 0]]
        else:
            padding = [[0, 0], [0, 0], [begin[0], end[0]], [begin[1], end[1]]]
        return tf.pad(inputs, padding)

    def max_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.max_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def average_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.average_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def dense_linear(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=None)

    def dense(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=self.activation)

    def activate(self, inputs, activation=None):
        activation = activation or self.activation
        return activation(inputs) if activation is not None else inputs

    def batch_norm(self, inputs, **kwargs):
        all_kwargs = dict(self.batch_norm_config)
        all_kwargs.update(kwargs)
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        bn_inputs = inputs
        outputs = tf.contrib.layers.batch_norm(
            inputs, is_training=self.training, data_format=data_format,
            fused=True, **all_kwargs)

        if ML_PERF_LOG:
            resnet_log_helper.log_batch_norm(
                input_tensor=bn_inputs, output_tensor=outputs, momentum=_BATCH_NORM_DECAY,
                epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training)
        return outputs

    def spatial_average2d(self, inputs):
        shape = inputs.get_shape().as_list()
        if self.data_format == 'channels_last':
            n, h, w, c = shape
        else:
            n, c, h, w = shape
        n = -1 if n is None else n
        x = tf.layers.average_pooling2d(inputs, (h, w), (1, 1),
                                        data_format=self.data_format)
        return tf.reshape(x, [n, c])

    def flatten2d(self, inputs):
        x = inputs
        if self.data_format != 'channels_last':
            # Note: This ensures the output order matches that of NHWC networks
            x = tf.transpose(x, [0, 2, 3, 1])
        input_shape = x.get_shape().as_list()
        num_inputs = 1
        for dim in input_shape[1:]:
            num_inputs *= dim
        return tf.reshape(x, [-1, num_inputs], name='flatten')

    def residual2d(self, inputs, network, units=None, scale=1.0, activate=False):
        outputs = network(inputs)
        c_axis = -1 if self.data_format == 'channels_last' else 1
        h_axis = 1 if self.data_format == 'channels_last' else 2
        w_axis = h_axis + 1
        ishape, oshape = [y.get_shape().as_list() for y in [inputs, outputs]]
        ichans, ochans = ishape[c_axis], oshape[c_axis]
        strides = ((ishape[h_axis] - 1) // oshape[h_axis] + 1,
                   (ishape[w_axis] - 1) // oshape[w_axis] + 1)
        with tf.name_scope('residual'):
            if (ochans != ichans or strides[0] != 1 or strides[1] != 1):
                inputs = self.conv2d_linear(inputs, units, 1, strides, 'SAME')
            x = inputs + scale * outputs
            if activate:
                x = self.activate(x)
        return x


def resnet_bottleneck_v1(builder, inputs, depth, depth_bottleneck, stride, filters,
                         basic=False):
    num_inputs = inputs.get_shape().as_list()[3]
    x = inputs
    with tf.name_scope('resnet_v1'):
        if ML_PERF_LOG:
            resnet_log_helper.log_begin_block(input_tensor=x, block_type=mlperf_log.BOTTLENECK_BLOCK)
        if depth == num_inputs:
            if stride == 1:#v1.5
                shortcut = x
            else:#v1
                shortcut = builder.max_pooling2d(x, 1, stride)
        else:
            shortcut = builder.conv2d_linear(x, depth, 1, stride, 'SAME')
            conv_input = x
            if ML_PERF_LOG:
                resnet_log_helper.log_conv2d(
            	input_tensor=conv_input, output_tensor=shortcut, stride=stride,
            	filters=filters*4, initializer=mlperf_log.TRUNCATED_NORMAL, use_bias=False)
                resnet_log_helper.log_projection(input_tensor=conv_input, output_tensor=shortcut)
        if basic:
            x = builder.pad2d(x, 1)
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'VALID')
            x = builder.conv2d_linear(x, depth, 3, 1, 'SAME')
        else:
            conv_input = x
            x = builder.conv2d(x, depth_bottleneck, 1, 1, 'SAME')
            if ML_PERF_LOG:
                resnet_log_helper.log_conv2d(
                    input_tensor=conv_input, output_tensor=x, stride=1,
                    filters=filters, initializer=mlperf_log.TRUNCATED_NORMAL, use_bias=False)
                mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
            conv_input = x
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
            
            if ML_PERF_LOG:
                resnet_log_helper.log_conv2d(
                    input_tensor=conv_input, output_tensor=x, stride=stride,
                    filters=filters, initializer=mlperf_log.TRUNCATED_NORMAL, use_bias=False)
                mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
            # x = builder.conv2d_linear(x, depth,            1, 1,      'SAME')
            conv_input = x
            x = builder.conv2d_linear_last_bn(x, depth, 1, 1, 'SAME')
            if ML_PERF_LOG:
                resnet_log_helper.log_conv2d(
                    input_tensor=conv_input, output_tensor=x, stride=1,
                    filters=filters*4, initializer=mlperf_log.TRUNCATED_NORMAL, use_bias=False)
                mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
                mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_SHORTCUT_ADD)
                mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
        x = tf.nn.relu(x + shortcut)
        if ML_PERF_LOG:
            resnet_log_helper.log_end_block(output_tensor=x)
        return x


def inference_resnet_v1_impl(builder, inputs, layer_counts, arch_type='C1+D', resnet_version='v1.5', basic=False): 
    x = inputs
    x = builder.conv2d(x, 64, 7, 2, 'SAME')
    num_filters=64

    x, argmax = tf.nn.max_pool_with_argmax( input=x, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME' )
    for i in range(layer_counts[0]):
        x = resnet_bottleneck_v1(builder, x, 256, 64, 1, num_filters, basic)
    for i in range(layer_counts[1]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v1(builder, x, 512, 128, 2 if i == 0 else 1, num_filters, basic)
    for i in range(layer_counts[2]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v1(builder, x, 1024, 256, 2 if i == 0 else 1, num_filters, basic)
    for i in range(layer_counts[3]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v1(builder, x, 2048, 512, 2 if i == 0 else 1, num_filters, basic)


   # x = builder.spatial_average2d(x)

    # same function as spatial average
    axis = [1,2]
    x = tf.reduce_mean( x, axis, keepdims=True )
    x = tf.reshape(x, [-1,2048])


    logits = tf.layers.dense(x, 1001, 
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=1))
    return logits

def inference_resnet_v1(config, inputs, nlayer, data_format='channels_last',
                        training=False, conv_initializer=None, bn_init_mode='adv_bn_init', bn_gamma_initial_value=1.0 ):

    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    if ML_PERF_LOG:
        mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_INITIAL_SHAPE,
                            value=inputs.shape.as_list()[1:])

    builder = LayerBuilder(tf.nn.relu, data_format, training, use_batch_norm=True,
                           conv_initializer=conv_initializer, bn_init_mode=bn_init_mode, bn_gamma_initial_value=bn_gamma_initial_value)
    if nlayer == 18:
        return inference_resnet_v1_impl(builder, inputs, [2, 2, 2, 2], basic=True)
    elif nlayer == 34:
        return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3], basic=True)
    elif nlayer == 50:
        return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3])
    elif nlayer == 101:
        return inference_resnet_v1_impl(builder, inputs, [3, 4, 23, 3])
    elif nlayer == 152:
        return inference_resnet_v1_impl(builder, inputs, [3, 8, 36, 3])
    else:
        raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                         nlayer)


