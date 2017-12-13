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

#from conceptnet5.language.lemmatize import lemmatize
#from lemmatize  import lemmatize
from nltk.stem.wordnet import WordNetLemmatizer

# STS specific relative file paths
tf.flags.DEFINE_string("sts_train_tsv", "data/17.train.tsv",
                       "tsv of train data")
tf.flags.DEFINE_string("text_data", "data/ftrain_dev_text.txt",
                       "text data")
tf.flags.DEFINE_string("label_data", "data/ftrain_dev_labels.txt",
                       "label data")
tf.flags.DEFINE_string("glove", "data/glove.twitter.27B.200d.txt",
                       "pretrained glove word embeddings.")

# allows for flags to be split across files, allowing specialized flags per .py
tf.flags.FLAGS._parse_flags()
# TODO perhaps make a flags.py with only tf.flags.DEFINE to handle this?
#   Or have the main only call single fucntions to run a model, and let it be
#   filled with flags.

FLAGS = tf.flags.FLAGS

#def read_data(ftext=FLAGS.text_data, flabels=FLAGS.label_data):
def read_data(ftext=FLAGS.text_data):
    with open(ftext, 'r') as dat_csv:
        csv_reader = csv.reader(dat_csv)

        pivot = []
        comparison = []
        feature =[]
        label = []

        lemmatize = WordNetLemmatizer().lemmatize # TODO use ConceptNet Lemmas

        for row in csv_reader:
            #pivot.append(lemmatize("en", row[0], "n")[0])
            #comparison.append(lemmatize("en", row[1], "n")[0])
            #feature.append(lemmatize("en", row[2], "n")[0])
            #label.append(lemmatize("en", row[3], "n")[0])

            #"""
            pivot.append(lemmatize(row[0]))
            comparison.append(lemmatize(row[1]))
            feature.append(lemmatize(row[2]))
            label.append(lemmatize(row[3]))
            """
            pivot.append(row[0])
            comparison.append(row[1])
            feature.append(row[2])
            label.append(row[3])
            #"""
        return (np.array(pivot),
               np.array(comparison),
               np.array(feature),
               np.array(label))

def embed_index(embed_dir=FLAGS.glove):
    embeddings_index = {}
    with open(embed_dir) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            values = line.split()
            word = values[0]
            idx = word.rfind("/")
            if idx != -1:
                word = word[(idx + 1):]
            if "_" not in word: # do not include ConceptNet's multi word embeds
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    return embeddings_index
