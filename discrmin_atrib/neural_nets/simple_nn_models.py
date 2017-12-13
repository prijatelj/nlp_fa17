"""
Several small and simple keras models for various things, mostly for learning
how to compare the sentences.

@author: Derek S. Prijatelj
"""
from __future__ import print_function

import os, errno, sys
import math
from pathos.multiprocessing import ProcessingPool as ThreadPool

import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import tensorflow as tf

import keras.backend as K
from keras import losses
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, LSTM, Dropout, \
    Conv1D, MaxPooling1D, UpSampling1D, \
    Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Reshape, Lambda, Flatten
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import RMSprop, Adam

#import sts_data_handler
#import embed_models

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

#Define constants for this file
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
K_FOLDS = FLAGS.kfolds
HIDDEN_NODES = FLAGS.hidden_nodes
HIDDEN_LAYERS= FLAGS.hidden_layers
TIME_STEPS=FLAGS.time_steps

if "threads" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("threads", 1, "Thread quantity for multiprocessing")

def constrained_categorical_crossentropy(ytrue, ypred):
  #ypred = K.clip(ypred, 0.0001, 0.99999)
  lim_zero = float(10**sys.float_info.min_10_exp)
  ypred = K.clip(ypred, lim_zero, 1.0-lim_zero)
  return losses.categorical_crossentropy(ytrue, ypred)

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
                                   activation='elu',
                                   name="Bi-LSTM_Hidden_" + identifier + "_"
                                        + str(i)),
                                   merge_mode="concat")(x)

        last = Bidirectional(LSTM(units,
                                  return_sequences=False,
                                  stateful=FLAGS.stateful,
                                  activation='elu',
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
                                   activation='elu',
                                   name="Bi-LSTM_Hidden_" + identifier + "_"
                                        + str(i)),
                                   merge_mode="concat")(x)
    #print("Hidden LSTM output = ", x.get_shape().as_list())
    return x

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
                     activation='elu',
                     name="LSTM_Hidden_" + identifier + "_" + str(i))(x)

            if FLAGS.dropout_rate != 0:
                x = Dropout(FLAGS.dropout_rate)(x)

        last = LSTM(units,
                    return_sequences=False,
                    stateful=FLAGS.stateful,
                    activation='elu',
                    name="LSTM_Hidden_" + identifier + "_" + str(time_step))
        x = last(x)

        #print("x_lstm_last shape = ", x.get_shape().as_list())
        #print("last.output_shape ", last.output_shape)
    else:
        for i in range(time_step):
            x = LSTM(units,
                     return_sequences=True,
                     stateful=FLAGS.stateful,
                     activation='elu',
                     name="LSTM_Hidden_" + identifier + "_" + str(i))(x)

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

    #"""
    if FLAGS.bidir:
        emb = bilstm_stack(emb, input_shape[-1], input_shape[0],
                                 identifier="1")
    else:
        emb = lstm_stack(emb,
                         time_step=input_shape[1]
                        )
    predictions = Dense(class_length,
                        activation='softmax',
                        name="Single_Dense")(emb)

    print("predictions.shape = ", predictions.get_shape().as_list())

    model = Model(input1, predictions)
    #opt = Adam()
    opt = Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=['accuracy', 'categorical_accuracy']
                 )
    return model

def dense(input_shape, embed_model, class_length=20):
    """
    """
    print("input_shape = ", input_shape, " with type = ", type(input_shape))
    input1 = Input(shape=[input_shape[1]])
    print("input1.shape = ", input1.get_shape().as_list())

    emb = embed_model(input1)
    print("\nemb shape = ", emb.get_shape().as_list(), "\n")

    for i in range(FLAGS.hidden_layers):
        emb = TimeDistributed(Dense(FLAGS.hidden_nodes, activation='elu'))(emb)
        #emb = Dense(FLAGS.hidden_nodes, activation='elu')(emb)
        if FLAGS.dropout_rate != 0:
            emb = Dropout(FLAGS.dropout_rate)(emb)

    emb = Flatten()(emb)
    predictions = Dense(#class_length,
                        2,
                        activation='softmax',
                        name="Single_Dense_Pred")(emb)

    print("predictions.shape = ", predictions.get_shape().as_list())

    model = Model(input1, predictions)
    #opt = Adam()
    opt = Adam(lr=FLAGS.learning_rate, epsilon=FLAGS.adam_epsilon)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  #loss=constrained_categorical_crossentropy,
                  metrics=['accuracy']
                 )
    return model

def conv(input_shape, embed_model, class_length=20):
    """
    """
    print("input_shape = ", input_shape, " with type = ", type(input_shape))
    input1 = Input(shape=[input_shape[1]])
    print("input1.shape = ", input1.get_shape().as_list())

    emb = embed_model(input1)
    print("\nemb shape = ", emb.get_shape().as_list(), "\n")

    conv = UpSampling1D(100)(emb)
    #conv = UpSampling2D((100,0))(emb)
    print("\nUpSampled shape = ", conv.get_shape().as_list(), "\n")

    conv = Conv1D(150, 25, activation="elu")(conv)
    conv = MaxPooling1D(2)(conv)
    conv = Conv1D(50, 5, activation="elu")(conv)
    conv = MaxPooling1D(2)(conv)

    for i in range(FLAGS.hidden_layers):
        conv = TimeDistributed(Dense(FLAGS.hidden_nodes, activation='elu'))(conv)
        if FLAGS.dropout_rate != 0:
            conv = Dropout(FLAGS.dropout_rate)(conv)

    conv = Flatten()(conv)
    predictions = Dense(class_length,
                        activation='sigmoid',
                        name="Single_Dense")(conv)

    print("predictions.shape = ", predictions.get_shape().as_list())

    model = Model(input1, predictions)
    #opt = RMSprop(lr=FLAGS.learning_rate)
    #opt = Adam()
    opt = Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=['accuracy']
                 )
    return model
