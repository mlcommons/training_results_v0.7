import tensorflow as tf
from data_generator import DataGenerator
from models import Model
import os,sys
import time
from mlperf_logging import mllog

from threading import Thread
###### npu ######
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu import util
# HCCL 
from npu_bridge.hccl import hccl_ops
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id
from npu_bridge.estimator.npu import npu_compile

from npu_bridge.helper import helper
gen_npu_ops = helper.get_gen_ops();
###### npu ######
rank_size = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID').split("-")[-1])
device_id = int(os.getenv('DEVICE_ID')) + rank_id * 8
###############################

# MLperf log
if device_id == 0:
	mllogger = mllog.get_mllogger()
	mllog.config(filename='resnet_close.log')
	mllog.config(
			default_namespace='worker1',
     	 default_stack_offset=1,
     	 default_clear_line=False,
     		root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__)))
			)
	mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value="resnet" )
	mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value="open" )
	mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="SIAT" )
	mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value="Ascend 910" )
	mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value="cloud" )
	mllogger.event(key=mllog.constants.CACHE_CLEAR )

params = { 
  #  'data_dir': '/opt/dataset/imagenet_TF',
  #  'data_dir': '/cache/ImageNet',
    'data_dir': '/home/work/datasets/Dataset',
    'num_classes': 1001,
    'batch_size': 32,
    'num_threads': 96,
    'use_synthetic': False,
    'model_name': 'resnet50',
    'data_format': 'channels_last',
    'arch_type': 'Original', 
    'resnet_version': 'v1.5',
    'ckpt_path': './results', 

    'learning_rate': 25.0,
    'momentum': 0.95,
    'weight_decay': 0.0001,
    'train_epochs': 24,
    'eval_interval_epochs':4,
    'print_interval':1, 

    # distributed config
    'use_lars': True,


    'dtype': tf.float32,
    # --- lr ----
    'mode':'cosine',
    'warmup_epochs': 7.6,




    # #### npu #####
    'iterations_per_loop_train': 10,
   
}
params['global_batch_size'] = rank_size * params['batch_size']

eval_graph = tf.Graph()

config = tf.ConfigProto()
config.allow_soft_placement = True

train_data_gen = DataGenerator(params)
eval_data_gen = DataGenerator(params)
###################### npu ##########################
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["enable_data_pre_proc"].b = True
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["hcom_parallel"].b = True
custom_op.parameter_map["min_group_size"].b = 1
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
custom_op.parameter_map["iterations_per_loop"].i = params['total_steps']
#custom_op.parameter_map["iterations_per_loop"].i = params['iterations_per_loop_train']
################ npu ###################


config.intra_op_parallelism_threads=96
config.inter_op_parallelism_threads=96



def main():
    main_graph = tf.Graph()

    model = Model(params)
    input_queues = []
    
    with main_graph.as_default():
        tf.set_random_seed(1)
        train_input_iterator = train_data_gen.make_iterator_initialize(training=True)
        print ('--- train make iterator -----')
        eval_input_iterator = eval_data_gen.make_iterator_initialize(training=False)
        print ('--- eval make iterator -----')
        train_input_init_op = train_input_iterator.initializer
        eval_input_init_op = eval_input_iterator.initializer
    
        train_sample = train_input_iterator.get_next()
        eval_sample = eval_input_iterator.get_next()
    
        # Flags to indicate train or eval      
        float_status = tf.constant([0.0], dtype=tf.float32)
        total_steps = tf.Variable(initial_value=tf.constant(0,tf.int32), trainable=False)
        train_steps = tf.train.get_or_create_global_step()
        total_steps = tf.assign_add(total_steps, 1)
        with tf.control_dependencies([total_steps]):
          eval_flag = tf.mod(total_steps - 1, params['total_steps_per_eval'] )
          init_local_flag = tf.equal( eval_flag, params['training_steps_between_evals']-1 )
    
 
        def train_fn():
            with tf.variable_scope('Res', reuse=False):
                train_op, predicted_label, base_loss, lr, training_s, labels = model.model_func(train_sample[0], train_sample[1], is_training=True, train_steps=train_steps)
              #  train_op, predicted_label, base_loss, labels = model.model_func(train_data, train_label, is_training=True)
                with tf.control_dependencies([train_op]):
                   # train_steps = tf.train.get_or_create_global_step()
                    increase_train_steps_op = tf.assign_add(train_steps, 1, name='NpuCompile')
                    with tf.control_dependencies([increase_train_steps_op]):
                        train_fn_op = tf.no_op(name='train_op_0')
            return train_fn_op, predicted_label, base_loss,lr, training_s, labels
        def eval_fn():
            with tf.variable_scope('Res', reuse=True):
                eval_op, predicted_label, base_loss, lr, training_s,  labels = model.model_func(eval_sample[0], eval_sample[1], is_training=False, train_steps=train_steps)
                with tf.control_dependencies([eval_op]):
                    eval_fn_op = tf.no_op(name='eval_op_0')
            return eval_fn_op, predicted_label, base_loss,lr, training_s, labels
    
    
        # choose to exe train or eval
        final_op, predicted_label, final_base_loss, lr, training_s, labels = tf.cond( eval_flag < params['training_steps_between_evals'], train_fn, eval_fn)
        with tf.control_dependencies([final_op]):
            final_op = tf.no_op(name='Final_op')
    
        # when eval, initial metric's local vars
        float_status = gen_npu_ops.npu_alloc_float_status() # when first step, avoid NaN
        weights = tf.greater(labels, -1)
        eval_value, metric_update_op = tf.metrics.accuracy( labels = labels, predictions=predicted_label, weights=weights )
        with tf.control_dependencies([metric_update_op]):
          #  local_float_status = gen_npu_ops.npu_get_float_status(float_status)
          #  cleared_float_status = gen_npu_ops.npu_clear_float_status(local_float_status)
          #  no_nan = tf.reduce_all( tf.equal( float_status, cleared_float_status ) )
          #  def allreduce_no_nan():
          #      return eval_accuracy
          #  def allreduce_nan():
          #      return tf.constant(0.0, tf.float32)
          #  eval_accuracy = tf.cond( no_nan, allreduce_no_nan, allreduce_nan )
            local_vars = tf.local_variables() # VAR total and count in Metric
            eval_accuracy = tf.divide(local_vars[0],local_vars[1])
            eval_accuracy = hccl_ops.allreduce( eval_accuracy, "sum", fusion=0 )
            print_op = tf.print(eval_accuracy, eval_flag, init_local_flag, total_steps, train_steps, local_vars[0], local_vars[1])
            with tf.control_dependencies([print_op]):
              print_op_2 = tf.identity( eval_accuracy )

        
        
        def clear_local_vars_true():
            clear_op_1 = tf.assign( local_vars[0], tf.constant(0, tf.float32) )
            clear_op_2 = tf.assign( local_vars[1], tf.constant(0, tf.float32) )
            with tf.control_dependencies([clear_op_1, clear_op_2]):
              clear_op = tf.no_op(name='clear_local_vars_true')
            return clear_op
        def clear_local_vars_false():
            clear_op = tf.no_op(name='clear_local_vars_false')
            return clear_op

        with tf.control_dependencies([print_op_2]):
            clear_op_final = tf.cond( init_local_flag, clear_local_vars_true, clear_local_vars_false )

        saver = tf.train.Saver()
    
    main_sess = tf.Session(graph=main_graph, config = config)
    
    
    with main_sess as sess:
            if device_id == 0:
                mllogger.start(key=mllog.constants.INIT_START)
                mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=params['global_batch_size'])
                mllogger.event(key="opt_name", value="lars")
                mllogger.event(key="lars_opt_weight_decay", value=params['weight_decay'])
                mllogger.event(key="lars_epsilon", value=0.0)
                mllogger.event(key="lars_opt_base_learning_rate", value=params['learning_rate'])
                mllogger.event(key="lars_opt_end_learning_rate", value=0.0001)
                mllogger.event(key="lars_opt_learning_rate_decay_poly_power", value=2)
                decay_steps = (params['train_epochs'] - params['warmup_epochs'])*params['training_steps_per_epoch']
                mllogger.event(key="lars_opt_learning_rate_decay_steps", value=decay_steps)
                mllogger.event(key="lars_opt_learning_rate_warmup_epochs", value=params['warmup_epochs'])
                mllogger.event(key="lars_opt_momentum", value=params['momentum'])

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            # ################# npu ##################
           # final_op = util.set_iteration_per_loop(sess, final_op, params['iterations_per_loop_train'])
            final_op = util.set_iteration_per_loop(sess, final_op, params['total_steps'])
            # ################# npu ##################
    
            time_start = time.time()
            real_steps = int( float(params['total_steps']) / float( params['iterations_per_loop_train'] ))
            
            # compile graph
            fetches = [final_op, clear_op_final,eval_accuracy, total_steps, train_steps, final_base_loss, eval_flag, lr, training_s]
            npu_compile.npu_compile( sess, fetches )
            sess.run(train_input_init_op)
            sess.run(eval_input_init_op)
            if device_id == 0:         
                mllogger.end(key=mllog.constants.INIT_STOP)
                mllogger.start(key=mllog.constants.RUN_START)
                mllogger.event(key='train_samples', value=params['num_training_samples'])
                mllogger.event(key='eval_samples', value=params['num_evaluate_samples'])
                       
            # start to train & eval
            sess.run( fetches )


def dequeue():
    global config
    tf.reset_default_graph()
    outfeed_log_tensors = npu_ops.outfeed_dequeue_op(
            channel_name="_npu_log",
            output_types=[tf.string],
            output_shapes=[()])
    with tf.Session() as sess:
      time_start = time.time()
      step_count = 0 
      current_epoch = 0 
      i = 0
      while True:
       # if step_count >= 0 :
        # cal flags
          current_block = (step_count // params['total_steps_per_eval']) + 1
          step_in_block = step_count % params['total_steps_per_eval']
          step_in_epoch = step_in_block % params['training_steps_per_epoch']
          epoch_in_block = step_in_block // params['training_steps_per_epoch'] + 1
          current_epoch = (current_block - 1) * params['eval_interval_epochs'] + epoch_in_block 


          time_end = time.time()
          time_duration = time_end - time_start
          fps = float(  params['global_batch_size'] ) / time_duration 
          time_start = time.time()
          result = sess.run( outfeed_log_tensors )
          if device_id == 0:
            # count block
            if step_in_block == 0:
                mllogger.start(key=mllog.constants.BLOCK_START, metadata={'first_epoch_num':current_epoch-1, 'epoch_count':params['eval_interval_epochs']})  
            if step_in_block == params['training_steps_between_evals']-1:
                mllogger.end(key=mllog.constants.BLOCK_STOP, metadata={'first_epoch_num':current_epoch-1} )  
           
            # count eval
          #  if step_in_block == params['training_steps_between_evals']-1:
          #      mllogger.start(key=mllog.constants.EVAL_START)  
            if step_in_block == params['total_steps_per_eval']-1:
                a = result[0].decode('UTF-8')
                acc = float( a.split(' ')[0] ) / float(rank_size)
           #     mllogger.end(key=mllog.constants.EVAL_STOP)  
                mllogger.event(key=mllog.constants.EVAL_ACCURACY, value = acc, metadata={'epoch_num': current_epoch - 1})  
                if acc > 0.759:
                    mllogger.end(key=mllog.constants.RUN_STOP)
                    break
 
            # finish running
            if step_count == params['total_steps']-1:
                mllogger.end(key=mllog.constants.RUN_STOP, metadata={'status':'success'})
                break

            # regular printings
            print ( '----LOGGING---- step:', step_count , ' epoch:', current_epoch, ' fps:', fps, ' time_duration(ms):', time_duration * 1000)

          step_count = step_count + 1

#            if (i % params['total_steps_per_eval']) == params['total_steps_per_eval'] - 1:
#                a = result[0].decode('UTF-8')
#                acc = float( a.split(' ')[0] ) / float(rank_size)
#                print ( '----Eval Result---- step:', i , ' fps:', fps, ' time_duration(ms):', time_duration * 1000, 'Accuracy:', acc, 'result:', result )
#                if acc > 0.759:
#                  print ('TIME_LOG: acheive 75.9% accuracy, training STOP,', time.time())
#            elif i % 1 == 0:
#                a = result[0].decode('UTF-8')
#                acc = float( a.split(' ')[0] ) / float(rank_size)
#                print ( '----Training---- step:', i , ' fps:', fps, ' time_duration(ms):', time_duration * 1000, 'Accuracy:', acc, 'result:', result)
#            if i == params['total_steps']-1:
#                print ('TIME_LOG: training End:', time.time())
#                break
#            if i == 0:
#                print ('TIME_LOG: training Start:', time.time())
#
#
#        i = i + 1
        

if __name__ == "__main__":
  t1 = Thread( target=dequeue )
  t1.start()
  main()

