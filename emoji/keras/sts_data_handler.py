"""
Data handler for the sts task. Loads the data in Keras, embeds the data based on
pretrained word vectors, and also splits the data in train and validate pairs
"""

from __future__ import print_function

import sys
import csv
import argparse
import numpy as np
from nltk import word_tokenize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard

# Purely for their FLAGs handling, will replace with personal argparse later.
import tensorflow as tf

# STS specific relative file paths
tf.flags.DEFINE_string("sts_train_tsv", "data/17.train.tsv",
                       "tsv of train data")
tf.flags.DEFINE_string("text_data", "data/ftext_dev_text.txt",
                       "text data")
tf.flags.DEFINE_string("label_data", "data/flabel_dev_labels.txt",
                       "label data")
tf.flags.DEFINE_string("glove", "data/glove.twitter.27B.200d.txt",
                       "pretrained glove word embeddings.")

# allows for flags to be split across files, allowing specialized flags per .py
tf.flags.FLAGS._parse_flags()
# TODO perhaps make a flags.py with only tf.flags.DEFINE to handle this?
#   Or have the main only call single fucntions to run a model, and let it be
#   filled with flags.

FLAGS = tf.flags.FLAGS

def read_data(ftext=FLAGS.text_data, flabels=FLAGS.label_data):
    with open(ftext, 'rb') as text, open(flabels, 'r') as labels:
        text_content = text.readlines()
        labels_content = labels.readlines()

        assert len(text_content) == len(labels_content)

        return np.array(text_content), np.array(labels_content)

def embed_index(glove_dir=FLAGS.glove):
    embeddings_index = {}
    with open(glove_dir) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index
