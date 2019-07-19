
import numpy as np
import tensorflow as tf
from tensorflow_train.layers.initializers import he_initializer, zeros_initializer
from tensorflow_train.utils.data_format import get_channel_index
from tensorflow_train.utils.print_utils import print_conv_parameters, print_upsample_parameters, print_shape_parameters, print_matwocaps_parameters, print_primary_matwocaps_parameters

debug_print_conv  =  True
debug_print_caps  =  True
debug_print_dense  =  True
debug_print_pool  =  False
debug_print_upsample  =  False
debug_print_others  =  False

def pad_for_conv(inputs, kernel_size, name, padding, data_format):
    if padding in ['symmetric', 'reflect']:
        # TODO check if this works for even kernels
        channel_index  =  get_channel_index(inputs, data_format)
        paddings  =  np.array([[0, 0]] + [[int(ks / 2)] * 2 for ks in kernel_size])
        paddings  =  np.insert(paddings, channel_index, [0, 0], axis = 0)
        outputs  =  tf.pad(inputs, paddings, mode = padding, name = name+'/pad')
        padding_for_conv  =  'valid'
    else:
        outputs  =  inputs
        padding_for_conv  =  padding
    return outputs, padding_for_conv

def conv2d(inputs,
           filters,
           kernel_size,
           name,
           activation = None,
           kernel_initializer = he_initializer,
           bias_initializer = zeros_initializer,
           normalization = None,
           is_training = False,
           data_format = 'channels_first',
           padding = 'same',
           strides = (1, 1),
           debug_print = debug_print_conv):
    node, padding_for_conv  =  pad_for_conv(inputs = inputs,
                                          kernel_size = kernel_size,
                                          name = name,
                                          padding = padding,
                                          data_format = data_format)
    outputs  =  tf.layers.conv2d(inputs = node,
                               filters = filters,
                               kernel_size = kernel_size,
                               name = name,
                               kernel_initializer = kernel_initializer,
                               bias_initializer = bias_initializer,
                               trainable = is_training,
                               data_format = data_format,
                               kernel_regularizer = tf.nn.l2_loss,
                               padding = padding_for_conv,
                               strides = strides)

    if normalization is not None:
        outputs  =  normalization(outputs, is_training = is_training, data_format = data_format, name = name+'/norm')

    if activation is not None:
        outputs  =  activation(outputs, name = name+'/activation')

    if debug_print:
        print_conv_parameters(inputs = inputs,
                              outputs = outputs,
                              kernel_size = kernel_size,
                              name = name,
                              activation = activation,
                              kernel_initializer = kernel_initializer,
                              bias_initializer = bias_initializer,
                              normalization = normalization,
                              is_training = is_training,
                              data_format = data_format,
                              padding = padding,
                              strides = strides)

    return outputs


def conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     name,
                     activation = None,
                     kernel_initializer = he_initializer,
                     bias_initializer = zeros_initializer,
                     normalization = None,
                     is_training = False,
                     data_format = 'channels_first',
                     padding = 'same',
                     strides = (1, 1),
                     debug_print = debug_print_conv):
    outputs  =  tf.layers.conv2d_transpose(inputs = inputs,
                                         filters = filters,
                                         kernel_size = kernel_size,
                                         name = name,
                                         kernel_initializer = kernel_initializer,
                                         bias_initializer = bias_initializer,
                                         trainable = is_training,
                                         data_format = data_format,
                                         kernel_regularizer = tf.nn.l2_loss,
                                         padding = padding,
                                         strides = strides)

    if normalization is not None:
        outputs  =  normalization(outputs, is_training = is_training, data_format = data_format, name = name+'/norm')

    if activation is not None:
        outputs  =  activation(outputs, name = name+'/activation')

    if debug_print:
        print_conv_parameters(inputs = inputs,
                              outputs = outputs,
                              kernel_size = kernel_size,
                              name = name,
                              activation = activation,
                              kernel_initializer = kernel_initializer,
                              bias_initializer = bias_initializer,
                              normalization = normalization,
                              is_training = is_training,
                              data_format = data_format,
                              padding = padding,
                              strides = strides)

    return outputs


def upsample2d(inputs, kernel_size, name = '', data_format = 'channels_first', debug_print = debug_print_upsample):
    outputs  =  tf.contrib.keras.layers.UpSampling2D(kernel_size, data_format = data_format, name = name)(inputs)
    if debug_print:
        print_upsample_parameters('nn',
                                  inputs,
                                  outputs,
                                  kernel_size,
                                  name,
                                  data_format,
                                  'same',
                                  kernel_size)
    return outputs


def avg_pool2d(inputs, kernel_size, strides = None, name = '', padding = 'same', data_format = 'channels_first', debug_print = debug_print_pool):
    if strides is None:
        strides  =  kernel_size
    outputs  =  tf.layers.average_pooling2d(inputs, kernel_size, strides, padding = 'same', data_format = data_format, name = name)
    if debug_print:
        print_pool_parameters(pool_type = 'avg',
                              inputs = inputs,
                              outputs = outputs,
                              kernel_size = kernel_size,
                              strides = strides,
                              name = name,
                              data_format = data_format,
                              padding = padding)
    return outputs


def max_pool2d(inputs, kernel_size, strides = None, name = '', padding = 'same', data_format = 'channels_first', debug_print = debug_print_pool):
    if strides is None:
        strides  =  kernel_size
    outputs  =  tf.layers.max_pooling2d(inputs, kernel_size, strides, padding = 'same', data_format = data_format, name = name)
    if debug_print:
        print_pool_parameters(pool_type = 'avg',
                              inputs = inputs,
                              outputs = outputs,
                              kernel_size = kernel_size,
                              strides = strides,
                              name = name,
                              data_format = data_format,
                              padding = padding)
    return outputs

def concat_channels(inputs, name = '', data_format = 'channels_first', debug_print = debug_print_others):
    axis  =  get_channel_index(inputs[0], data_format)
    outputs  =  tf.concat(inputs, axis = axis, name = name)
    if debug_print:
        print_shape_parameters(inputs, outputs, name, 'concat')
    return outputs




########################################################################

################################
#       UTILS-FUNCTION         #
################################

def caps_length( x, axis = 2):
    pred = tf.sqrt(tf.reduce_sum(tf.square(x), axis = axis, keepdims = True))
    pred = tf.squeeze(pred,axis = axis)
    return pred

def caps_duallength(x,pos_dim, app_dim, axis = 2):
    x_mat,x_mat2 = tf.split(x, [np.product(pos_dim),np.product(app_dim)], axis = axis)
    pred  =  caps_length(x_mat, axis = axis) *  caps_length(x_mat2, axis = axis)
    return pred

def squash(p,axis = -1):
      p_norm_sq  =  tf.reduce_sum(tf.square(p), axis = axis, keepdims = True)
      p_norm  =  tf.sqrt(p_norm_sq + 1e-9)
      v  =  p_norm_sq / (1. + p_norm_sq) * p / p_norm
      return v

def Psquash(p,axis = -1):
    v = (p)/tf.reduce_max(tf.abs(p),axis = axis, keepdims = True)
    return v

def mesh2d(shape):
    x,y = np.meshgrid(range(shape[2]),range(shape[1]))
    return  y.flatten(),x.flatten()

def coordinate_addition(b,shape):
  y,x = mesh2d(shape)
  
  coord_add = np.zeros((shape[1]*shape[2], 1, shape[-2], shape[-1]),dtype = np.float32)
  coord_add[:,0,-1,0] += x
  coord_add[:,0,-1,1] += y
  coord_add[:,0,-1,0] /= shape[2].value
  coord_add[:,0,-1,1] /= shape[1].value

  b += (coord_add)
  b = tf.reshape(b,shape)
  return b

def matmult2d(a,b):
    mat = []
    for i in range(a.get_shape()[-2]):
        mat.append(tf.multiply(tf.expand_dims(tf.gather(a,i,axis = -2),axis = -1),b))
    c = tf.reduce_sum(tf.stack(mat,axis = -3),axis = -2)
    return c

def l2_normalize_dim(b,axis):
  denom = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(b),axis = axis)),axis = axis)
  b /= denom
  return b

################################
#          ROUTING             #
################################

def routing2d(routing, t_0, u_hat_t_list):

  N, z_1, H_1, W_1, o, t_1 = u_hat_t_list.get_shape().as_list()

  c_t_list  =  []
  b  =  tf.zeros([N, H_1, W_1, t_0,t_1])
  b_t_list  =  [tf.squeeze(b_t, axis = -1) for b_t in tf.split(b, t_1, axis = -1)]
  
  u_hat_t_list_ =  [tf.squeeze(u_hat_t, axis = -1) for u_hat_t in tf.split(u_hat_t_list, t_1, axis = -1)] 
  for d in range(routing):

    r_t_mul_u_hat_t_list  =  []
    for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
      r_t = tf.nn.softmax(b_t,axis = -1)

      if d < routing-1 :
        r_t  =  tf.expand_dims(r_t, axis = 1) # [N, 1, H_1, W_1, t_0]
        r_t_mul_u_hat_t_list.append(tf.reduce_sum(r_t * u_hat_t,axis = -1))  #sum along the capsule to form the output

      else :
        c_t_list.append(r_t)

    if d < routing-1 :  
      p  =  r_t_mul_u_hat_t_list 
      v = squash(p,axis = 2)

      b_t_list_  =  []
      idx = 0

      for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
        v_t1 = tf.reshape(tf.gather(v, [idx], axis = 0),[N,z_1,H_1,W_1,1])

        #Evaluate agreement
        rout = tf.reduce_sum(v_t1*u_hat_t)
        b_t_list_.append(b_t + rout)
        idx += 1

      b_t_list  =  b_t_list_
  return c_t_list


def dual_routing2d(routing, t_0, u_hat_t_list, z_pos, z_app):
  N, z_1, H_1, W_1, o, t_1 = u_hat_t_list.get_shape().as_list()

  c_t_list  =  []
  b  =  tf.zeros([N, H_1, W_1, t_0,t_1])
  b_t_list  =  [tf.squeeze(b_t, axis = -1) for b_t in tf.split(b, t_1, axis = -1)]

  u_hat_t_list_ =  [tf.squeeze(u_hat_t, axis = -1) for u_hat_t in tf.split(u_hat_t_list, t_1, axis = -1)]  
  for d in range(routing):
    r_t_mul_u_hat_t_list  =  []

    for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
      r_t = tf.nn.sigmoid(b_t)

      if d < routing-1 :
        r_t  =  tf.expand_dims(r_t, axis = 1) # [N, 1, H_1, W_1, t_0]
        r_t_mul_u_hat_t_list.append(tf.reduce_sum(r_t * u_hat_t,axis = -1))  #sum along the capsule to form the output

      else :
        c_t_list.append(r_t)

    if d < routing-1 :  
      p  =  r_t_mul_u_hat_t_list
      p_pos, p_app = tf.split(p, [z_pos,z_app], axis = 2)
      v_app = squash(p_app,axis = 2)
      v_pos = Psquash(p_pos,axis = 2)

      b_t_list_  =  []
      idx = 0
      for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):

        u_hat_pos, u_hat_app = tf.split(u_hat_t, [z_pos,z_app], axis = 1)       
        v_t1_pos = tf.reshape(tf.gather(v_pos, [idx], axis = 0),[N,z_pos,H_1,W_1,1])
        v_t1_app = tf.reshape(tf.gather(v_app, [idx], axis = 0),[N,z_app,H_1,W_1,1])

        #Evaluate agreement
        rout = tf.reduce_sum(u_hat_pos*v_t1_pos,axis = 1)  * tf.reduce_sum(u_hat_app * v_t1_app,axis = 1) 
        b_t_list_.append(b_t + rout)
        idx += 1

      b_t_list  =  b_t_list_

  return c_t_list


################################
#          CAPSULE             #
################################


def primary_caps2d_matwo(inputs, capsule_types, pos_dim, app_dim, op, kernel_size, strides, name, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu, padding = 'SAME',  is_training = False, data_format = "channels_last", debug_print = debug_print_caps):
    """
      inputs : Input with shape [batch/N, channel/z_0, height/H_0, width/W_0] 
      op : "conv" or "deconv"
      kernel_size : Kernel size of (de)convolution and routing  = > k
      strides : Stride size of (de)convotluion
      capsule_type : The number of types of target capsule  = > t
      pos_dim : The [Y,X] dimension of the matrix representing the pose
      app_dim : The [Y,X] dimension of the matrix representing the appearance
    """

    with tf.variable_scope(name):
      t_0  =  1
      t_1  =  capsule_types
      z_app = np.product(app_dim)
      z_pos = np.product(pos_dim)
      N, z_0, H_0, W_0 = inputs.get_shape().as_list()

      # Create the appearance projection matrix
      ones_kernel = tf.ones([1,1,1,1])
      mult_app_all  =  tf.layers.conv2d(ones_kernel, t_0*t_1*app_dim[1]*app_dim[1], 1, 1, kernel_initializer = kernel_initializer,trainable = is_training,padding = padding, use_bias = False, name = 'mult_app')
      mult_app = tf.reshape(mult_app_all,[t_1, app_dim[1], app_dim[1]])

      # Extract appearance value
      if op  ==  "conv":
        u_spat_t  =  tf.layers.conv2d(inputs, z_app * capsule_types, kernel_size = kernel_size, strides = strides, kernel_initializer = kernel_initializer,trainable = is_training, activation = activation, padding = padding, use_bias = True,data_format = data_format, name = 'spatial_k')
      elif op  ==  "deconv":
        u_spat_t  =  tf.layers.conv2d_transpose(inputs, z_app * capsule_types, ḱernel_size = kernel_size, strides = strides, kernel_initializer = kernel_initializer,trainable = is_training, activation = activation, padding = padding, use_bias = True,data_format = data_format, name = 'spatial_k')
      H_1  =  u_spat_t.get_shape()[2]
      W_1  =  u_spat_t.get_shape()[3]
      u_t_app =  tf.transpose(u_spat_t, (0,2,3,1)) # [N, t_1 * z_app * z_pos, H_1, W_1] => [N, H_1, W_1, t_1 * z_app * z_pos]

      # Initialize the pose matrix with identity
      u_t_pos = tf.cast(np.zeros([N , H_1, W_1, t_1,pos_dim[0],pos_dim[1]]),dtype = tf.float32)
      u_t_pos  =  tf.reshape(u_t_pos,[N * H_1 * W_1, t_1, pos_dim[0], pos_dim[1]])
      identity = tf.eye(pos_dim[1])
      identity = identity[pos_dim[1]-pos_dim[0]:,:]
      u_t_pos += identity
      u_hat_t_pos  =  tf.reshape(u_t_pos , [N, H_1, W_1, t_1, z_pos])

      # Apply the matrix multiplication to the appearance matrix
      u_t_app  =  tf.reshape(u_t_app,[N, H_1 , W_1, t_1, app_dim[0], app_dim[1]])
      u_t_app = matmult2d(u_t_app, mult_app)
      u_hat_t_app  =  tf.reshape(u_t_app , [N, H_1, W_1, t_1, z_app])

      # Squash the appearance matrix (Psquashing the pose won't change it)
      v_pos = u_hat_t_pos
      v_app = squash(u_hat_t_app)

      v = tf.concat([v_pos,v_app],axis = -1)
      outputs = tf.transpose(v,(0,3,4,1,2)) #[t_1, N, H_1, W_1, z_1] => [N, t, z , H, W] #[N, H_1, W_1, t_1, z_1] => [N, t, z , H_1, W_1]

    if debug_print:
        print_primary_matwocaps_parameters(inputs = inputs,
                              outputs = outputs,
                              capsule_types = capsule_types,
                              app_dim = app_dim,
                              pos_dim = pos_dim,
                              kernel_size = kernel_size,
                              activation = activation,
                              name = name,
                              kernel_initializer = kernel_initializer,
                              is_training = is_training,
                              padding = padding,
                              strides = strides)


    return outputs


def caps2d_matwo(inputs, capsule_types, pos_dim, app_dim, routing, routing_type, op, kernel_size, strides, name, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), coord_add = True, padding = 'SAME',  is_training = False, debug_print = debug_print_caps):
    """
      inputs : Input with shape [batch/N, capsule_type/t_0, channel/z_0, height/H_0, width/W_0] 
      op : "conv" or "deconv"
      kernel_size : Kernel size of (de)convolution and routing  = > k
      strides : Stride size of (de)convotluion
      capsule_type : The number of types of target capsule  = > t
      pos_dim : The [Y,X] dimension of the matrix representing the pose
      app_dim : The [Y,X] dimension of the matrix representing the appearance
      coord_add : If the kernel center is added to the pose matrix
      routing : The number of routing operation
      routing_type : Routing type (dual routing ("dual") or original routing ("dynamic"))
    """
    
    with tf.variable_scope(name):
      t_1  =  capsule_types
      z_app = np.product(app_dim)
      z_pos = np.product(pos_dim)
      N, t_0, z_0, H_0, W_0  =  inputs.get_shape().as_list() #tf.shape(u)
      
      # Initialize weight
      ones_kernel = tf.ones([1,1,1,1])

      mult_pos_all  =  tf.layers.conv2d(ones_kernel, t_0*t_1*pos_dim[1]*pos_dim[1], 1, 1,kernel_initializer = kernel_initializer,trainable = is_training,padding = padding,use_bias = False,name = 'mult_pos')
      mult_pos_all = tf.reshape(mult_pos_all,[t_0,pos_dim[1]*pos_dim[1],t_1])

      mult_app_all  =  tf.layers.conv2d(ones_kernel, t_0*t_1*app_dim[1]*app_dim[1], 1, 1,kernel_initializer = kernel_initializer,trainable = is_training,padding = padding,use_bias = False,name = 'mult_app')
      mult_app_all = tf.reshape(mult_app_all,[t_0, app_dim[1]*app_dim[1] ,t_1])

      bias_app = tf.layers.conv2d(ones_kernel, t_0*t_1, 1, 1,kernel_initializer=kernel_initializer,trainable=is_training,padding=padding,use_bias=False,name='bias_app')
      bias_app=tf.reshape(bias_app,[t_0, t_1])

      # For each child (input) capsule (t_0) project into all parent (output) capsule domain (t_1)
      idx_all = 0
      u_hat_t_list  =  []
      
      u_t_list  =  [tf.squeeze(u_t, axis = 1) for u_t in tf.split(inputs, t_0, axis = 1)]
      for u_t in u_t_list: 
        u_t = tf.reshape(u_t,[N*z_0,H_0,W_0,1])

        # Apply spatial kernel
        if op  ==  "conv":
          u_spat_t  =  tf.layers.conv2d(u_t, t_1, kernel_size, strides, kernel_initializer = tf.initializers.variance_scaling(distribution = 'uniform'), trainable = is_training, padding = padding, use_bias = False)
        elif op  ==  "deconv":
          u_spat_t  =  tf.layers.conv2d_transpose(u_t, t_1, kernel_size, strides, kernel_initializer = tf.initializers.variance_scaling(distribution = 'uniform'),trainable = is_training, padding = padding,use_bias = False)
        else:
          raise ValueError("Wrong type of operation for capsule")
        # Some shape operation
        H_1  =  u_spat_t.get_shape()[1]
        W_1  =  u_spat_t.get_shape()[2]
        u_spat_t  =  tf.reshape(u_spat_t, [N , z_0, H_1, W_1, t_1])
        u_spat_t  =  tf.transpose(u_spat_t, (0,2,3,4,1))
        u_spat_t  =  tf.reshape(u_spat_t, [N , H_1, W_1, t_1*z_0])
        u_t_pos, u_t_app = tf.split(u_spat_t, [t_1*z_pos,t_1*z_app], axis = -1)
        u_t_pos  =  tf.reshape(u_t_pos,[N , H_1 , W_1, t_1, pos_dim[0], pos_dim[1]])
        u_t_app  =  tf.reshape(u_t_app,[N , H_1 , W_1, t_1, app_dim[0], app_dim[1]])

        # Gather projection matrices and bias
        mult_pos = tf.gather(mult_pos_all,idx_all,axis = 0)
        mult_pos = tf.reshape(mult_pos,[t_1,pos_dim[1],pos_dim[1]])
        mult_app = tf.gather(mult_app_all,idx_all,axis = 0)
        mult_app = tf.reshape(mult_app,[t_1, app_dim[1], app_dim[1]])
        bias = tf.reshape(tf.gather(bias_app,idx_all,axis=0),(1,1,1,t_1, 1, 1))

        u_t_app += bias

        # Prepare the pose projection matrix
        mult_pos = l2_normalize_dim(mult_pos,axis = -2)
        if coord_add:
          mult_pos = coordinate_addition(mult_pos,[1 , H_1 , W_1, t_1, pos_dim[1], pos_dim[1]])

        u_t_pos = matmult2d(u_t_pos,mult_pos)
        u_t_app = matmult2d(u_t_app,mult_app)

        # Store the result
        u_hat_t_pos  =  tf.reshape(u_t_pos , [N, H_1, W_1, t_1, z_pos])
        u_hat_t_app  =  tf.reshape(u_t_app , [N, H_1, W_1, t_1, z_app])
        u_hat_t = tf.concat([u_hat_t_pos,u_hat_t_app],axis = -1)
        u_hat_t_list.append(u_hat_t)

        idx_all += 1

      u_hat_t_list = tf.stack(u_hat_t_list,axis = -2)
      u_hat_t_list = tf.transpose(u_hat_t_list,[0,5,1,2,4,3]) #[N, H, W, t_1, t_0, z]  = > [N, z, H_1, W_1, t_0, t_1]

      # Routing operation
      if routing>0:
        if routing_type is 'dynamic':
          if type(routing) is list:
            routing = routing[-1]
          c_t_list = routing2d(routing = routing, t_0 = t_0, u_hat_t_list = u_hat_t_list) #[T1][N,H,W,to]
        elif routing_type is 'dual':
          if type(routing) is list:
            routing = routing[-1]
          c_t_list = dual_routing2d(routing = routing, t_0 = t_0, u_hat_t_list = u_hat_t_list, z_app = z_app, z_pos = z_pos) #[T1][N,H,W,to]
        else:
          raise ValueError(routing_type+' is an invalid routing; try dynamic or dual')
      else:
        routing_type = 'NONE'
        c = tf.ones([N, H_1, W_1, t_0, t_1])
        c_t_list  =  [tf.squeeze(c_t, axis = -1) for c_t in tf.split(c, t_1, axis = -1)]

      #Form each parent capsule through the weighted sum of all child capsules
      r_t_mul_u_hat_t_list  =  []
      u_hat_t_list_ =  [(tf.squeeze(u_hat_t, axis = -1)) for u_hat_t in tf.split(u_hat_t_list, t_1, axis = -1)]
      for c_t, u_hat_t in zip(c_t_list, u_hat_t_list_): 
        r_t  =  tf.expand_dims(c_t, axis = 1)
        r_t_mul_u_hat_t_list.append(tf.reduce_sum(r_t * u_hat_t,axis = -1))
      
      p  =  r_t_mul_u_hat_t_list 
      p = tf.stack( p, axis = 1)
      p_pos, p_app = tf.split(p, [z_pos,z_app], axis = 2)

      #Squash the weighted sum to form the final parent capsule
      v_pos = Psquash(p_pos,axis = 2)
      v_app = squash(p_app,axis = 2)

      outputs = tf.concat([v_pos,v_app],axis = 2)

    if debug_print:
        print_matwocaps_parameters(inputs = inputs,
                              outputs = outputs,
                              routing = routing,
                              coord_add = coord_add,
                              routing_type = routing_type,
                              capsule_types = capsule_types,
                              app_dim = app_dim,
                              pos_dim = pos_dim,
                              kernel_size = kernel_size,
                              name = name,
                              kernel_initializer = kernel_initializer,
                              is_training = is_training,
                              padding = padding,
                              strides = strides)

    return outputs

