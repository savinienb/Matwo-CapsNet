
    ######################################################################################
    #               Adapted from https://github.com/lalonderodney/SegCaps                #
    # Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241) #
    ######################################################################################


import tensorflow as tf
from tensorflow.keras import layers, models

from LungSeg.SegCaps.capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Length
from tensorflow_train.layers.layers import  caps_length

from keras import backend as K
K.set_image_data_format('channels_last')

def SegCaps_multilabels(input_shape, is_training, num_labels, data_format):

    with tf.variable_scope('caps',reuse=tf.AUTO_REUSE):
        n_class=num_labels
        padding='same'
        activation=tf.nn.relu
        input_shape=tf.transpose(input_shape,(0,2,3,1))

        # Layer 1: Just a conventional Conv2D layer
        #conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
        conv1 = tf.layers.conv2d(input_shape, filters=32, kernel_size=5, name='conv1', padding='same', kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.nn.relu, trainable=is_training)

        # Reshape layer to be 1 capsule x [filters] atoms
        _, H, W, C = conv1.get_shape()
        H=H.value
        W=W.value
        C=C.value
        conv1_reshaped = layers.Reshape((H, W, 1, C))(conv1)

        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=6, num_atoms=32, strides=2, padding='same',
                                        routings=1, name='primarycaps',is_training=is_training)(conv1_reshaped)

        # Layer 2: Convolutional Capsule
        conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=6, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_2_1',is_training=is_training)(primary_caps)

        # Layer 2: Convolutional Capsule
        conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=6, num_atoms=32, strides=2, padding='same',
                                        routings=3, name='conv_cap_2_2',is_training=is_training)(conv_cap_2_1)

        # Layer 3: Convolutional Capsule
        conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_3_1',is_training=is_training)(conv_cap_2_2)

        # Layer 3: Convolutional Capsule
        conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                        routings=3, name='conv_cap_3_2',is_training=is_training)(conv_cap_3_1)

        # Layer 4: Convolutional Capsule
        conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_4_1',is_training=is_training)(conv_cap_3_2)

        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_1_1',is_training=is_training)(conv_cap_4_1)

        # Skip connection
        #up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])
        up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=6, num_atoms=32, strides=1,
                                          padding='same', routings=3, name='deconv_cap_1_2',is_training=is_training)(up_1)

        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=6, num_atoms=32, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_2_1',is_training=is_training)(deconv_cap_1_2)

        # Skip connection
        up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=6, num_atoms=32, strides=1,
                                          padding='same', routings=3, name='deconv_cap_2_2',is_training=is_training)(up_2)

        # Layer 3 Up: Deconvolutional Capsule
        deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=6, num_atoms=32, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_3_1',is_training=is_training)(deconv_cap_2_2)

        # Skip connection
        up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

        # Layer 4: Convolutional Capsule: 1x1
        x = ConvCapsuleLayer(kernel_size=1, num_capsule=num_labels, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='seg_caps',is_training=is_training)(up_3)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        #out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)
        x=tf.transpose(x,(0,3,4,1,2))
        out_seg=caps_length(x,axis=2)

        return out_seg
