"""
Main file for running Keras models

@author: Derek S. Prijatelj
"""
from __future__ import print_function

import os, errno
import math
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as ThreadPool

import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import tensorflow as tf

from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, LSTM
from keras.layers.core import Reshape, Lambda, Flatten
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical

#import sts_data_handler
#import embed_models

# Flags for basic model hyper parameters
if "epochs" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("epochs", 1,
                            "The number of epochs for a training run")
if "epoch_steps" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("epoch_steps", 1,
                            "The number of epoch_steps for an epoch run")
if "batch_size" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("batch_size", 1,
                            "The number of records to process per batch during "
                            + "training.")
if "learning_rate" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate of optimizer")
if "kfolds" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("kfolds", 10,
                            "The number of folds for k-fold cross entropy.")

# Flags determining the specifics of parts of the model
if "hidden_nodes" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("hidden_nodes", 200,
                            "The number of folds for k-fold cross entropy.")
if "hidden_layers" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("hidden_layers", 25,
                            "The number of folds for k-fold cross entropy.")
if "dropout_rate" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_float("dropout_rate", 0.0,
                            "Dropout rates probability.")

# For LSTMs
if "time_steps" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("time_steps", 1,
                            "The number of time_steps of input sequences.")
if "sequences" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("sequences", False,
                            "Hidden LSTMs return sequences if true")
if "stateful" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("stateful", False,
                            "Hidden LSTMs return stateful if true")
if "bidir" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("bidir", False,
                            "LSTM's are bi-directional if True")

# Flags that determine the model being run
if "model" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("model", "dense",
                           "The name of the simple_nn_model to run")
if "embedding" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("embedding", "glove",
                           "The name of the simple_nn_model to run")
if "comparator" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("comparator", "perceptron",
                           "The name of the comparator to used to compare "
                           + "embeddings")
if "train_embed" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("train_embed", False,
                            "Trains embeddings on data if True")
if "model_id" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("model_id", "", "The ID of the model")

if "results_dir" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("results_dir", "results/keras/",
                           "The name of the simple_nn_model to run")
if "threads" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("threads", 1, "Thread quantity for multiprocessing")

# Data:
if "sts_test_tsv" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("sts_test_tsv", "data/17.test.tsv",
                           "tsv of test data")
if "test" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("test", False,
                            "Trains on all data and evaluates on test data if "
                            + " True")

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

# flag dependencies...
#from simple_nn_models import perceptron, lstm

#Define constants for this file
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
K_FOLDS = FLAGS.kfolds
HIDDEN_NODES = FLAGS.hidden_nodes
TIME_STEPS=FLAGS.time_steps
MODEL_ID=FLAGS.model_id

def only_train(model, train_data, train_labels,
               test_data=None, test_labels=None):
    tb_callback = TensorBoard(log_dir='./Graph',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    if test_data is None or test_labels is None:
        history = model.fit(train_data, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=[tb_callback])
    else:
         history = model.fit(train_data, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=[tb_callback],
                  validation_data=(test_data,
                                   test_labels))

    return model, history

def only_eval(model, test_data, test_labels):
    eval_results = model.evaluate(test_data,
                                  test_labels)
    pred = np.squeeze(model.predict(test_data,
                                    batch_size=BATCH_SIZE))
    #corr, _ = pearsonr(pred, test_labels)

    #for i, single_prediction in enumerate(pred):
    #    print("{:<20} should = {:}".format(single_prediction, test_labels[i]))
    #    if i == 25:
    #        break

    #eval_results += [corr]
    eval_metrics = model.metrics_names #+ ["pearsonr"]
    results_dict = dict(zip(eval_metrics, eval_results))

    return results_dict

def train_and_eval(model, train_data, train_labels, test_data, test_labels):
    model, history = only_train(
        model, train_data, train_labels, test_data, test_labels)

    results_dict = history.history
    for k, v in results_dict.items():
        results_dict[k] = v[-1]

    print()
    for key, value in results_dict.items():
        print(key, " = ", value)

    return results_dict, model

if __name__ == "__main__":
    tf.app.run()
