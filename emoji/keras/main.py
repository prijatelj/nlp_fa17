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

import keras.backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, LSTM
from keras.layers.core import Reshape, Lambda, Flatten
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import RMSprop

import sts_data_handler
import embed_models
import sif_embed

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
    tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of optimizer")
if "kfolds" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("kfolds", 10,
                            "The number of folds for k-fold cross entropy.")

# Flags determining the specifics of parts of the model
if "hidden_nodes" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_integer("hidden_nodes", 300,
                            "The number of folds for k-fold cross entropy.")

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
    tf.flags.DEFINE_string("model", "perceptron",
                           "The name of the simple_nn_model to run")
if "embedding" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_string("embedding", "arora",
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
from simple_nn_models import perceptron, perceptron2, lstm
from mv_lstm import mv_lstm

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
        model.fit(train_data, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=[tb_callback],
                  validation_data=(test_data,
                                   test_labels))
    else:
        model.fit(train_data, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=[tb_callback])

    return model

def only_eval(model, test_data, test_labels):
    eval_results = model.evaluate(test_data,
                                  test_labels)
    pred = np.squeeze(model.predict(test_data,
                                    batch_size=BATCH_SIZE))
    corr, _ = pearsonr(pred, test_labels)

    for i, single_prediction in enumerate(pred):
        print("{:<20} should = {:}".format(single_prediction, test_labels[i]))
        if i == 25:
            break

    eval_results += [corr]
    eval_metrics = model.metrics_names + ["pearsonr"]
    results_dict = dict(zip(eval_metrics, eval_results))

    return results_dict

def train_and_eval(model, train_data, train_labels, test_data, test_labels):
    model = only_train(model, train_data, train_labels, test_data, test_labels)
    print("\nIn-Sample Evaluation:")
    in_sample_results = only_eval(model, train_data, train_labels)
    print("\nOut-of-Sample Evaluation:")
    results_dict = only_eval(model, test_data, test_labels)

    # change in-sample dict key names to avoid overwriting
    for key in in_sample_results.keys():
        in_sample_results["IS_" + key] = in_sample_results.pop(key)
    results_dict.update(in_sample_results)

    print()
    for key, value in results_dict.iteritems():
        print(key, " = ", value)

    return results_dict, model

def k_fold_multiprocess((i, (train, test), data, labels, data_shape,
                        create_model, tmp_save_dir)):
    """
    Helper function for multiprocess to properly detuple agruments
    """
    return k_fold_process(i, (train, test), data, labels, data_shape,
                          create_model, tmp_save_dir)

def k_fold_process(i, (train, test), data, labels, data_shape, create_model,
                   tmp_save_dir):
    """
    processes a single loop of kfolds
    """
    print("Running Fold", i+1, "/", K_FOLDS)
    model = None # Clearing the NN.
    model = create_model(data_shape)
    model_name = create_model.__name__

    if model_name == "perceptron":
        train_data = data[train]
        test_data = data[test]
    else:
        train_data = [data[train, 0, :], data[train, 1, :]]
        test_data = [data[test, 0, :], data[test, 1, :]]


    print("data type = ", type(data))
    print("labels type = ", type(labels))
    print("train type = ", type(train))

    results, trained_model = train_and_eval(model,
                       train_data, labels[train],
                       test_data, labels[test])
                       #steps=(math.ceil(len(train)/BATCH_SIZE),
                       #       math.ceil(len(test)/BATCH_SIZE)))

    saved_model_dir, saved_model_file = create_logs(model_name)
    saved_model_file = saved_model_file[:-3] + "h5py"

    print("saved_mode_file: " , saved_model_file)
    print("saved_mode_file/ ",
          saved_model_file[:(saved_model_file.rfind("/")+1)])
    try:
        os.makedirs(saved_model_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    trained_model.save(saved_model_file)

    tmp_save_file = str(i) + "Fold.csv"
    try:
        os.makedirs(tmp_save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(tmp_save_dir + tmp_save_file, 'w') as tmp:
        for i, (key, value) in enumerate(results.iteritems()):
            tmp.write(key + "," + str(value))
            if i < len(results) - 1:
                tmp.write("\n")
    return results

def k_fold(data, labels, create_model, data_shape=None):
    """
    performs k-fold cross-entropy on provided data and model.

    @param data: data to be split by k-fold cross entropy
    @param labels: labels/targets of data
    @param create_model: pointer to function that creates model
    @param data_shape: shape of data for passing to create_model, or a pre-built
        model that is to be built upon further
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True)
    total_eval = []

    print("length data = ", len(data))
    print("length labels = ", len(labels))

    model_name = create_model.__name__

    _, tmp_save_dir = create_logs(model_name)
    tmp_save_dir = "/tmp/" + tmp_save_dir[:-4] + "/"

    if FLAGS.threads <= 1:
        for i, (train, test) in enumerate(skf.split(data, labels)):
            total_eval.append(k_fold_process(i, (train, test), data, labels,
                data_shape, create_model, tmp_save_dir))
    else:
        k_fold_args = zip(
                         range(K_FOLDS),
                         skf.split(data,labels),
                         [data] * K_FOLDS,
                         [labels] * K_FOLDS,
                         [data_shape] * K_FOLDS,
                         [create_model] * K_FOLDS,
                         [tmp_save_dir] * K_FOLDS
                         )
        pool = ThreadPool(FLAGS.threads)
        total_eval = pool.map(k_fold_multiprocess, k_fold_args)
        pool.close()
        pool.join()

    values = []
    for results in total_eval:
        values.append(results.values())
    mean = np.mean(values, axis=0)
    results_dict = dict(zip(total_eval[0].keys(), mean))

    log_dir, log_file = create_logs(model_name)

    print("\nOverall Metrics: " + log_dir + log_file)
    for key, value in results_dict.iteritems():
        print(key, " = ", value)

    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(log_dir + log_file, 'w') as log:
        for i, (key, value) in enumerate(results_dict.iteritems()):
            log.write(key + "," + str(value))
            if i < len(results_dict) - 1:
                log.write("\n")

def final_train_test(create_model, train_data, train_labels, data_shape,
                     test_data, test_labels):
    data = np.append(train_data, test_data, axis=0)
    labels = np.append(train_labels, test_labels, axis=0)
    train = range(len(train_data))
    test = range(len(train_data), len(train_data) + len(test_data))
    model_name = create_model.__name__

    # create temporary save directory
    _, tmp_save_dir = create_logs(model_name)
    tmp_save_dir = "/tmp/" + tmp_save_dir[:-4] + "/"

    print("Training and Testing has begun.")
    results = k_fold_process(0, (train, test), data, labels, data_shape,
                             create_model, tmp_save_dir)

    log_dir, log_file = create_logs(model_name)

    print("\nOverall Metrics: " + log_dir + log_file)
    for key, value in results.iteritems():
        print(key, " = ", value)

    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(log_dir + log_file, 'w') as log:
        for i, (key, value) in enumerate(results.iteritems()):
            log.write(key + "," + str(value))
            if i < len(results) - 1:
                log.write("\n")

def create_logs(model_name, true_test=FLAGS.test):
    log_dir = FLAGS.results_dir

    if true_test:
        log_dir += "Test/"

    log_dir +=  model_name + "/" + FLAGS.embedding + "/"

    if not true_test:
        log_dir += str(K_FOLDS) + "_folds/"

    log_dir += str(EPOCHS) + "_epochs/" + str(FLAGS.learning_rate) \
               + "_learning_rate/"

    if FLAGS.train_embed:
        log_dir += "embeddings_trained/"

    if model_name == "lstm":
        if FLAGS.sequences:
            log_dir += "sequences/"
        if FLAGS.stateful:
            log_dir += "stateful/"
        if FLAGS.bidir:
            log_dir += "bidirectional/"

    if model_name == "mv_lstm":
        log_dir += FLAGS.sim_func + "/" + str(FLAGS.kpools) + "_pools/"

    log_file =  str(HIDDEN_NODES) + "dense_" +  str(TIME_STEPS) \
        + "timesteps_" + MODEL_ID + "_" \
        + str(datetime.now()).replace(' ', "-at-").replace(":", "-") + ".csv"

    return log_dir, log_file

def main(argv):
    # Embed Data based on FLAGS
    if FLAGS.embedding == "arora" and "perceptron" in FLAGS.model:
        if FLAGS.test:
            try:
                embedded_sents = np.load("data/embed_test_train_arora.npy")
                rates = np.load("data/test_train_rates.npy")
                test_seq = np.load("data/embed_test_arora.npy")
                test_rates = np.load("data/test_rates.npy")
            except IOError as ioex:
                print(ioex)
                print("saving files")
                sif_embed.embed(True)
                embedded_sents = np.load("data/embed_test_train_arora.npy")
                rates = np.load("data/test_train_rates.npy")
                test_seq = np.load("data/embed_test_arora.npy")
                test_rates = np.load("data/test_rates.npy")
        else:
            try:
                embedded_sents = np.load("data/embed_arora.npy")
                rates = np.load("data/rates.npy")
            except IOError as ioex:
                print(ioex)
                print("saving files")
                sif_embed.embed()
                embedded_sents = np.load("data/embed_arora.npy")
                rates = np.load("data/rates.npy")
    elif FLAGS.embedding == "keras_tokenizer" \
            or "perceptron" not in FLAGS.model:
        sent_pairs, rates = sts_data_handler.read_tsv()
        embed_index = sts_data_handler.embed_index()

        if FLAGS.test:
            # Must tokenize together, and separate at end
            test_pairs, test_rates = sts_data_handler.read_tsv(
                FLAGS.sts_test_tsv)
            #test_embed_index = sts_data_handler.embed_index()

            train_examples = len(sent_pairs)
            sent_pairs = np.vstack((sent_pairs, test_pairs))
            #embed_index.update(test_embed_index)

        embed_model, pad_seq = embed_models.word_embed_tokenizer(sent_pairs,
                                                                 embed_index)
        zipped = zip(pad_seq[0], pad_seq[1])
        embedded_sents = np.array(zipped)

        if FLAGS.test:
            test_seq = embedded_sents[train_examples :]
            embedded_sents = embedded_sents[: train_examples]

    print("embedded_sents.shape = ", embedded_sents.shape)

    if FLAGS.test:
        # Select model to run STS task with embedding
        if FLAGS.model == "perceptron":
            final_train_test(perceptron, embedded_sents, rates,
                             embedded_sents.shape[1:], test_seq, test_rates)
        elif FLAGS.model == "perceptron2":
            final_train_test(perceptron2, embedded_sents, rates,
                             embedded_sents.shape[2:], test_seq, test_rates)
        elif FLAGS.model == "lstm":
            final_train_test(lstm, embedded_sents, rates,
                             ([embedded_sents.shape[2]], embed_model),
                             test_seq, test_rates)
        elif FLAGS.model == "mv_lstm":
            final_train_test(mv_lstm, embedded_sents, rates,
                             ([embedded_sents.shape[2]], embed_model),
                             test_seq, test_rates)
    else:
        # Select model to run STS task with embedding
        if FLAGS.model == "perceptron":
            k_fold(embedded_sents, rates, perceptron, embedded_sents.shape[1:])
        elif FLAGS.model == "perceptron2":
            k_fold(embedded_sents, rates, perceptron2, embedded_sents.shape[2:])
        elif FLAGS.model == "lstm":
            k_fold(embedded_sents, rates, lstm,
                   ([embedded_sents.shape[2]], embed_model))
        elif FLAGS.model == "mv_lstm":
            k_fold(embedded_sents, rates, mv_lstm,
                   ([embedded_sents.shape[2]], embed_model))

if __name__ == "__main__":
    tf.app.run()
