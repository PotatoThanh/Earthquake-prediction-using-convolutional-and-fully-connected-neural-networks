import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model

def normal_model(input_shape, number_output):
    with tf.name_scope('input'):
        inputs = Input(input_shape)
        net = Lambda(lambda x: K.squeeze(x, axis=2))(inputs)
    with tf.name_scope('Dense_100'):
        net = Dense(units=100, activation='relu')(net)
        net = Dropout(0.5)(net)
    with tf.name_scope('Dense_50'):
        net = Dense(units=50, activation='relu')(net)
        net = Dropout(0.4)(net)
    with tf.name_scope('Dense_10'):
        net = Dense(units=10, activation='relu')(net)
        net = Dropout(0.3)(net)
    with tf.name_scope('output'):
        outputs = Dense(units=number_output)(net)

    return Model(inputs=inputs, outputs=outputs)

def CNN_model(input_shape, number_output):
    with tf.name_scope('input'):
        inputs = Input(input_shape)
    with tf.name_scope('Conv1D'):
        net = Conv1D(64, 2, activation='relu')(inputs)
        net = MaxPooling1D()(net)
        net = Conv1D(32, 2, activation='relu')(net)
        net = MaxPooling1D()(net)
        net = Flatten()(net)

    with tf.name_scope('Dense_150'):
        net = Dense(units=150, activation='relu')(net)
        net = Dropout(0.2)(net)
    with tf.name_scope('Dense_100'):
        net = Dense(units=100, activation='relu')(net)
        net = Dropout(0.2)(net)
    with tf.name_scope('Dense_50'):
        net = Dense(units=50, activation='relu')(net)
        net = Dropout(0.1)(net)
    with tf.name_scope('output'):
        outputs = Dense(units=number_output)(net)

    return Model(inputs=inputs, outputs=outputs)

