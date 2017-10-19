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
tf.flags.DEFINE_string("glove", "data/glove.840B.300d.txt",
                       "pretrained glove word embeddings.")

# allows for flags to be split across files, allowing specialized flags per .py
tf.flags.FLAGS._parse_flags()
# TODO perhaps make a flags.py with only tf.flags.DEFINE to handle this?
#   Or have the main only call single fucntions to run a model, and let it be
#   filled with flags.

FLAGS = tf.flags.FLAGS

def read_tsv_old(filename=FLAGS.sts_train_tsv):
    with open(filename, 'rb') as tsv:
        read_tsv = csv.reader(tsv, delimiter='\t', quoting=csv.QUOTE_NONE)
        # Trading off memory for speed
        sentence_dict = {}

        # meta data variables
        max_word_count = 0
        rates = []
        sent_pairs = []
        sent_lengths = []

        #for rate, sent1, sent2 in read_tsv:
        for rate, sent1, sent2 in read_tsv:
            if sent1 not in sentence_dict:
                sentence_dict[sent1] = word_tokenize(sent1)
                # TODO split on spaces, convert to lower case?
                # Punctuation is its own token!
                length = len(sentence_dict[sent1])
                if length > max_word_count:
                    max_word_count = length

            if sent2 not in sentence_dict:
                sentence_dict[sent2] = word_tokenize(sent2)
                length = len(sentence_dict[sent2])
                if length > max_word_count:
                    max_word_count = length

            rates.append(float(rate))
            sent_pairs.append([sentence_dict[sent1], sentence_dict[sent2]])
            #print(sentence_dict[sent1])
            #print(sentence_dict[sent2])

        rates = np.asarray(rates)
        sent_pairs = np.array(sent_pairs)

        #print(type(sent_pairs[0, 1]))
        print("sent_pairs shape: ", sent_pairs.shape)

    return rates, sent_pairs, sent_lengths, max_word_count

def read_tsv(filename=FLAGS.sts_train_tsv):
    with open(filename, 'rb') as tsv:
        read_tsv = csv.reader(tsv, delimiter='\t', quoting=csv.QUOTE_NONE)

        # meta data variables
        rates = []
        #sents1 = []
        #sents2 = []
        sent_pairs = []

        for rate, sent1, sent2 in read_tsv:
            rates.append(rate)
            #sents1.append(sent1)
            #sents2.append(sent2)
            sent_pairs.append([sent1, sent2])

    #return sents1, sents2, rates
    #return sent_pairs, rates
    return np.array(sent_pairs), np.array(rates, dtype=np.float32)

def embed_index(glove_dir=FLAGS.glove):
    embeddings_index = {}
    with open(glove_dir) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index
