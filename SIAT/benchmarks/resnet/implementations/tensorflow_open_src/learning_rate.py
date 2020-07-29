import tensorflow as tf
import numpy as np


def get_lr(params, global_step):
  lr_each_step = []
  if params['mode'] == 'steps':
    lr_max = params['learning_rate']
    boundry_epoch = [ 30 , 60, 80 ]
    boundaries = [ i * params['training_steps_per_epoch'] for i in boundry_epoch ]

    print ('lr boundaries:', boundaries)
    total_steps = int( params['training_steps_per_epoch'] * params['train_epochs'] )
    print ('lr total steps:', total_steps)
    for i in range(total_steps):
        if i < boundaries[0]:
            lr = lr_max
        elif i < boundaries[1]:
            lr = lr_max * 0.1
        elif i < boundaries[2]:
            lr = lr_max * 0.01
        else:
            lr = lr_max * 0.001
        lr_each_step.append(lr)

  elif params['mode'] == 'cosine':
    total_steps = int( params['training_steps_per_epoch'] * params['train_epochs'] )
    warmup_steps = int( params['training_steps_per_epoch'] * params['warmup_epochs'] )
    lr_max = params['learning_rate']
    for i in range( total_steps+10000 ):
      if i < warmup_steps:
        lr = cos_warmup_1980( i, warmup_steps, lr_max, warmup_lr = 0.8 )
      elif i <= total_steps:
        lr = cos_decay_1980( i, warmup_steps, total_steps, lr_max )
      else:
        lr = 0.0
      lr_each_step.append(lr)
    

  current_step = global_step
  print ('generated_learning_rate:',lr_each_step)
  lr_each_step = tf.convert_to_tensor( lr_each_step )
  learning_rate = tf.gather( lr_each_step, current_step )

  return learning_rate


def cos_warmup_1980(  global_step, warmup_steps, max_lr, warmup_lr ):
    PI = 3.14159265359
    ang = PI +  PI * ( float(global_step+1) / float(warmup_steps) )
    offset  = (max_lr - warmup_lr )* 0.5*( 1.0 + np.cos( ang ) )
    res = warmup_lr + offset
    return res

def cos_decay_1980(  global_step, warmup_steps, total_steps, max_lr ):
    PI = 3.14159265359
    ang =  PI * ( float(global_step - warmup_steps+1) / float(total_steps - warmup_steps) )
    offset  = max_lr * 0.5*( 1.0 + np.cos( ang ) )
    return offset
