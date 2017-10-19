"""
Matrix-Vector-LSTM (MV-LSTM) based on Want et. al 2016
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

import sif_embed
import sts_data_handler
import embed_models
from simple_nn_models import bilstm_stack
from NTN.neural_tensor_layer import NeuralTensorLayer

if "sim_func" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("sim_func", "tensor", "Similarity function used.")
if "kpools" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("kpools", 5, "k Number of pools for pooling.")

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

#Define constants for this file
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
K_FOLDS = FLAGS.kfolds
HIDDEN_NODES = FLAGS.hidden_nodes
TIME_STEPS = FLAGS.time_steps

def k_max_pool(x, k, pool_size, strides=(1,1)):
    #pools = K.pool2d(x, pool_size, strides)
    pools = K.pool2d(x, pool_size, strides)
    return tf.nn.top_k(pools, k, sorted=True)

def reduced_shape(shapes):
    shape1, shape2 = shapes

    return shape1
    return (shape1[0], 1)

def euclidean_dist(tensors, axis=1):
    x, y = tensors
    return K.sqrt(K.sum(K.square(y - x), axis=axis))
    #return K.sqrt(K.sum(K.square(y - x), axis=axis, keepdims=True))

def cosine_dist(tensors, axis=1):
    x, y = tensors
    x = K.l2_normalize(x, axis=axis)
    y = K.l2_normalize(y, axis=axis)
    return -K.mean(x * y, axis=axis)
    #return -K.mean(x * y, axis=axis, keepdims=True)

def bilinear(tensors, input_shape, bias=None):
    x, y = tensors
    identity = tf.constant(np.eye(input_shape[-1]), dtype=tf.float32)
    #TODO Need to figure out how to do element wise
    x = tf.reshape(x, input_shape[:-1]+[1])
    if bias is None:
        return K.transpose(x) * identity * y # this needs to be elment wise
    else:
        return K.transpose(x) * identity * y + bias

def mv_lstm((input_shape, embed_models_tup)):
    """
    Keras Implementation of the Matrix-Vector-LSTM by Wan et. al 2016.
    """
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    #if FLAGS.stateful:
        #input1 = Input(shape=input_shape, batch_shape=[BATCH_SIZE]+input_shape)
        #input2 = Input(shape=input_shape, batch_shape=[BATCH_SIZE]+input_shape)

    print("input1.shape = ", input1.get_shape().as_list())

    (embed_model1, embed_model2)= embed_models_tup

    emb1 = embed_model1(input1)
    emb2 = embed_model2(input2)

    print("\nemb1 shape = ", emb1.get_shape().as_list(), "\n")

    # single bidirectional lstm orig. allow differnt tests
    sent_emb1 = bilstm_stack(emb1, input_shape[-1], 1, True, identifier="1")
    sent_emb2 = bilstm_stack(emb2, input_shape[-1], 1, True, identifier="2")

    print("sent_emb1 shape = ", sent_emb1.get_shape().as_list())
    print("sent_emb2 shape = ", sent_emb2.get_shape().as_list())

    if FLAGS.sim_func == "tensor":
        # Neural Tensor Network
        print("input_shape[1]", input_shape[-1])

        mv_out = NeuralTensorLayer(input_shape[-1],
                                   input_dim=input_shape[-1]*2
                                  )([sent_emb1, sent_emb2])

        print("mv_out.shape = ", mv_out.get_shape().as_list())
        sim_tensor = mv_out #TODO remove tmp fix when k_max_pool works

        # k Max Pool
        #sim_tensor = Lambda(k_max_pool,
        #                     arguments={"k":FLAGS.kpools, "pool_size":(3,3)}
        #                    )(mv_out)
    elif FLAGS.sim_func == "L2" or FLAGS.sim_func == "euclidean":
        sim_tensor = Lambda(euclidean_dist)([sent_emb1,
                                                            sent_emb2])
    elif FLAGS.sim_func == "cos" or FLAGS.sim_func == "cosine":
        sim_tensor = Lambda(cosine_dist)([sent_emb1, sent_emb2])

    print("sim_tensor output shape = ", sim_tensor.get_shape().as_list())
    #print("sim_tensor size = ", np.prod(sim_tensor.get_shape().as_list()))

    # MLP As many initial Perceptron layers as there are pools?
    #   TODO Then perceptron all?
    dense = Dense(FLAGS.kpools, # match the output dim of top_kpools exactily
                  activation='elu',
                  kernel_initializer='he_normal')(sim_tensor)
    dense = Dense(int(math.ceil(FLAGS.kpools/2)),
                  activation='elu',
                  kernel_initializer='he_normal')(dense)
    dense = Dense(int(math.ceil(FLAGS.kpools/4)),
                  activation='elu',
                  kernel_initializer='he_normal')(dense)

    print("dense shape = ", dense.get_shape().as_list())

    predictions = Dense(1, activation='linear', name="Final_Dense")(dense)

    model = Model([input1, input2], predictions)
    opt = RMSprop(lr=FLAGS.learning_rate, clipvalue=5)
    model.compile(optimizer=opt,#'rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy',
                           'mean_squared_error',
                           'mean_absolute_error',
                           'mean_absolute_percentage_error'
                          ])
    return model
