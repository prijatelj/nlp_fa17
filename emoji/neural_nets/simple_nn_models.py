"""
Several small and simple keras models for various things, mostly for learning
how to compare the sentences.

@author: Derek S. Prijatelj
"""
from __future__ import print_function

import os, errno
import math
from pathos.multiprocessing import ProcessingPool as ThreadPool

import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import tensorflow as tf

import keras.backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, LSTM
from keras.layers.core import Reshape, Lambda, Flatten
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import RMSprop

#import sts_data_handler
#import embed_models

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

#Define constants for this file
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
K_FOLDS = FLAGS.kfolds
HIDDEN_NODES = FLAGS.hidden_nodes
TIME_STEPS=FLAGS.time_steps


def perceptron(input_shape):
    inputs = Input(shape=(input_shape))
    print(inputs.get_shape().as_list())

    dense = Dense(HIDDEN_NODES,
                  activation='softmax',
                  kernel_initializer="he_normal")

    x = dense(inputs)

    #TODO 2nd 300d weights are all initialized to zero, need to fix or make new
    print("Dense weights len, len[0] = ", len(dense.get_weights()),
          len(dense.get_weights()[0]))
    print("Dense weights = ", dense.get_weights())

    x = Reshape([HIDDEN_NODES * 2])(x)

    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy',
                           'mean_squared_error',
                           'mean_absolute_error',
                           'mean_absolute_percentage_error'
                          ])
    return model

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True),
                            K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def lstm_stack(x,
                units=HIDDEN_NODES,
                time_step=TIME_STEPS,
                sequences=FLAGS.sequences,
                identifier=""):
    """
    easy function call for creating multiple LSTMs in a stack or sequence
    """
    if not sequences:
        for i in range(0, time_step-1):
            x = LSTM(units,
                     return_sequences=True,
                     stateful=FLAGS.stateful,
                     activation='softmax',
                     name="LSTM_Hidden_" + identifier + "_" + str(i))(x)

        last = LSTM(units,
                    return_sequences=False,
                    stateful=FLAGS.stateful,
                    activation='softmax',
                    name="LSTM_Hidden_" + identifier + "_" + str(time_step))
        x = last(x)

        #print("x_lstm_last shape = ", x.get_shape().as_list())
        #print("last.output_shape ", last.output_shape)
    else:
        for i in range(time_step):
            x = LSTM(units,
                     return_sequences=True,
                     stateful=FLAGS.stateful,
                     activation='softmax',
                     name="LSTM_Hidden_" + identifier + "_" + str(i))(x)

    #print("Hidden LSTM output = ", x.get_shape().as_list())
    return x

def bilstm_stack(x,
                 units=HIDDEN_NODES,
                 time_step=TIME_STEPS,
                 sequences=FLAGS.sequences,
                 stateful=FLAGS.stateful,
                 identifier=""):
    """
    easy function call for creating multiple bidirectional LSTMs in a stack or
    sequence.
    """
    if not sequences:
        for i in range(0, time_step-1):
            x = Bidirectional(LSTM(units,
                                   return_sequences=True,
                                   stateful=FLAGS.stateful,
                                   activation='softmax',
                                   name="Bi-LSTM_Hidden_" + identifier + "_"
                                        + str(i)),
                                   merge_mode="concat")(x)

        last = Bidirectional(LSTM(units,
                                  return_sequences=False,
                                  stateful=FLAGS.stateful,
                                  activation='softmax',
                                  name="Bi-LSTM_Hidden_" + identifier + "_"
                                       + str(time_step)),
                             merge_mode="concat")
        x = last(x)

        #print("x_lstm_last shape = ", x.get_shape().as_list())
        #print("last.output_shape ", last.output_shape)
    else:
        for i in range(time_step):
            x = Bidirectional(LSTM(units,
                                   return_sequences=True,
                                   stateful=FLAGS.stateful,
                                   activation='softmax',
                                   name="Bi-LSTM_Hidden_" + identifier + "_"
                                        + str(i)),
                                   merge_mode="concat")(x)
    #print("Hidden LSTM output = ", x.get_shape().as_list())
    return x

def lstm(input_shape, embed_model, class_length=20):
    """
    basic LSTM model that recieves two sentences and embeds them as words and
    learns their relation.
    """
    print("input_shape = ", input_shape, " with type = ", type(input_shape))

    input1 = Input(shape=[input_shape[1]])
    print("input1.shape = ", input1.get_shape().as_list())

    emb = embed_model(input1)
    print("\nemb shape = ", emb.get_shape().as_list(), "\n")

    if FLAGS.bidir:
        sent_emb = bilstm_stack(emb, input_shape[-1], input_shape[0],
                                 identifier="1")
    else:
        sent_emb = lstm_stack(emb, identifier="1",
                              units=1,
                              time_step=input_shape[1]
                             )

    predictions = Dense(class_length, activation='softmax', name="Single_Dense")(sent_emb)

    print("predictions.shape = ", predictions.get_shape().as_list())

    model = Model(input1, predictions)
    opt = RMSprop(lr=FLAGS.learning_rate)
    model.compile(optimizer=opt,#'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           'mean_squared_error',
                           'mean_absolute_error',
                           'mean_absolute_percentage_error'
                          ])
    return model
