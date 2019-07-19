
import tensorflow as tf
from tensorflow_train.utils.data_format import get_image_axes, get_channel_index
import numpy as np

def softmax(labels, logits, weights=None,  data_format='channels_first'):
    channel_index = get_channel_index(labels, data_format)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=channel_index)
    return tf.reduce_mean(loss)


def spread_loss(labels,logits,m_low=0.2,m_hight=0.9,iteration_low_to_high=100000,global_step=100000,data_format='channels_first'):
    m=m_low+(m_hight-m_low)*tf.minimum(tf.to_float(global_step/iteration_low_to_high),tf.to_float(1))
    n_labels=labels.get_shape()[1]
    labels=tf.transpose(labels,(1, 0, 2, 3))
    logits=tf.transpose(logits,(1, 0, 2, 3))
    labels=tf.manip.reshape(labels,[n_labels,-1])  
    logits=tf.manip.reshape(logits,[n_labels,-1])

    true_class_logits=tf.reduce_max(labels*logits,axis=0)
    margin_loss_pixel_class=tf.square(tf.nn.relu((m-true_class_logits+logits)*(1-labels)))

    loss=tf.reduce_mean(tf.reduce_sum(margin_loss_pixel_class,axis=0))

    return loss


def weighted_spread_loss(labels,logits,m_low=0.2,m_hight=0.9,iteration_low_to_high=100000,global_step=100000, data_format='channels_first'):
  w_l=np.array([0.00705479,0.03312549,0.02664785,0.4437354 ,0.44254721,0.04688926]) * 6
  channel_index = get_channel_index(labels, data_format)
  
  m=m_low+(m_hight-m_low)*tf.minimum(tf.to_float(global_step/iteration_low_to_high),tf.to_float(1))
  n_labels=labels.get_shape()[1]
  labels=tf.transpose(labels,[1, 0, 2, 3])
  logits=tf.transpose(logits,[1, 0, 2, 3])
  labels=tf.manip.reshape(labels,[n_labels,-1])  
  logits=tf.manip.reshape(logits,[n_labels,-1])  

  true_class_logits=tf.reduce_max(labels*logits,axis=0)
  margin_loss_pixel_class=tf.square(tf.nn.relu((m-true_class_logits+logits)*tf.abs(labels-1)))

  loss=[]
  for i in range(len(w_l)):
    loss.append(w_l[i]*margin_loss_pixel_class*tf.gather(labels,[i],axis= channel_index))

  loss=tf.reduce_sum(tf.stack(loss,axis=0),axis=0)
  loss=tf.reduce_mean(tf.reduce_sum(loss,axis=0))

  return loss

def weighted_softmax(labels, logits, data_format='channels_first'):
  w_l=np.array([0.00705479,0.03312549,0.02664785,0.4437354 ,0.44254721,0.04688926]) * 6
  channel_index = get_channel_index(labels, data_format)
  loss_s=tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits, dim = channel_index)

  loss=[]
  for i in range(len(w_l)):
    loss.append(w_l[i] * loss_s * tf.gather(labels,[i],axis = channel_index))

  loss=tf.reduce_sum(tf.stack(loss,axis=0),axis=0)
  return tf.reduce_mean(loss)
