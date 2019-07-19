
import tensorflow as tf
from tensorflow_train.layers.layers import  caps_length, caps_duallength, caps2d_matwo, primary_caps2d_matwo
from tensorflow_train.layers.initializers import he_initializer


def Matwo_CapsNet(input, num_labels, is_training,  routing_type, routing=3, data_format='channels_last'):

    padding = 'SAME'
    coord_add = True
    routing = 3
    pos_dim = [4,4]
    app_dim= [5,5]
    level_caps = [5,5,6,7]

    prediction =Caps(input=input, num_labels=num_labels, is_training=is_training, routing_type=routing_type, routing=routing, pos_dim=pos_dim, app_dim=app_dim, coord_add=coord_add, level_caps=level_caps,  padding=padding, data_format=data_format)
    return prediction

def MatVec_CapsNet(input, num_labels, is_training,  routing_type, routing=3, data_format='channels_last'):

    padding = 'SAME'
    coord_add = True
    routing = 3
    pos_dim = [4,4]
    app_dim= [1,5]
    level_caps = [10,5,6,7]

    prediction = Caps(input=input, num_labels=num_labels, is_training=is_training, routing_type=routing_type, routing=routing, pos_dim=pos_dim, app_dim=app_dim, coord_add=coord_add, level_caps=level_caps, padding=padding, data_format=data_format)
    return prediction


def Caps(input, num_labels, is_training, routing_type, routing, pos_dim, app_dim, coord_add, level_caps,  padding,  data_format):

    input_dim=len(input.get_shape())
    if input_dim==5:
        input=tf.squeeze(input,axis=1)
        input=tf.transpose(input,[1,0,2,3])
    
    x=primary_caps2d_matwo(input, pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[0]),kernel_size=5, strides=1, name='primary_caps', padding=padding,op="conv",is_training=is_training,data_format=data_format)
    
    x=caps2d_matwo(x,routing=1, routing_type=routing_type, pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[1]),kernel_size=5,name='caps_1df_cd1',coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)
    skip1 = x
            # 1/2
    x=caps2d_matwo(x,routing=routing, routing_type=routing_type, pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[1]),kernel_size=5,name='caps_12_cd1',coord_add=coord_add, padding=padding,strides=2,op="conv",is_training=is_training)

    x=caps2d_matwo(x,routing=routing, routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[1]),kernel_size=5,name='caps_12_cd2', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)
    skip2 = x

    # 1/4
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[2]),kernel_size=5, name='caps_14_cd1', coord_add=coord_add, padding=padding,strides=2,op="conv",is_training=is_training)
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[2]),kernel_size=5, name='caps_14_cd2', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)
    skip3 = x

    # 1/8
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[3]),kernel_size=5,name='caps_18_cd1', coord_add=coord_add, padding=padding,strides=2,op="conv",is_training=is_training)
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[3]),kernel_size=5,name='caps_18_cd2', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[3]),kernel_size=5,name='caps_18_cd3', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)

    # 1/4
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[3]),kernel_size=4,name='caps_14_du1', coord_add=coord_add, padding=padding,strides=2,op="deconv",is_training=is_training)
    x = tf.concat([x, skip3], axis=1)
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[3]),kernel_size=5,name='caps_14_cu2', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)

    # 1/2
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[2]),kernel_size=4,name='caps_12_du1', coord_add=coord_add, padding=padding,strides=2,op="deconv",is_training=is_training)
    x = tf.concat([x, skip2], axis=1)
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[2]),kernel_size=5,name='caps_12_cu2', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)
    
    # 1
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=int(level_caps[2]),kernel_size=4,name='caps_1_du1', coord_add=coord_add, padding=padding,strides=2,op="deconv",is_training=is_training)
    x = tf.concat([x, skip1], axis=1)
   
    
    x=caps2d_matwo(x,routing=routing,routing_type=routing_type,pos_dim=pos_dim,app_dim=app_dim,capsule_types=num_labels,kernel_size=1,name='caps_1_c2', coord_add=coord_add, padding=padding,strides=1,op="conv",is_training=is_training)

    if routing_type is 'dual':
        prediction = caps_duallength(x,pos_dim=pos_dim,app_dim=app_dim)
    else:
        prediction = caps_length(x)

    return prediction
