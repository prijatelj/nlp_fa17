"""
Simple LSTM to solve the STS en-en task.
"""

from __future__ import print_function

import os
import argparse
import numpy as np
from nltk import word_tokenize, RegexpTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard

import sts_data_handler

# allows for flags to be split across files, allowing specialized flags per .py
tf.flags.FLAGS._parse_flags()
# TODO perhaps make a flags.py with only tf.flags.DEFINE to handle this?
#   Or have the main only call single fucntions to run a model, and let it be
#   filled with flags.

FLAGS = tf.flags.FLAGS

def prepare_embed_matrix():
    # prepare embedding matrix
    num_words = min(FLAGS.max_words, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return  embedding_layer

def main(argv):
    rates, sent_pairs, snet_lengths, max_word_count = read_tsv()
    sent_pairs, word_index = word_embed(sent_pairs)
    # TODO finish implementing train_valid_split
    train, valid = train_valid_split(rates, sent_pairs, sent_lengths)

    # TODO obviously finish the rest of the model

    prepare_embed_matrix()

    tbCallBack = TensorBoard(log_dir='./Graph',
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)
    return

if __name__ == "__main__":
    main(sys.argv)
