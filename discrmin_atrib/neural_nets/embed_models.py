from __future__ import print_function

import os
import sys
import csv
import numpy as np
from nltk import word_tokenize, RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, concatenate, Embedding
#from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.callbacks import TensorBoard

import tensorflow as tf

#if "train_embed" not in tf.flags.FLAGS.__dict__['__flags']:
#    tf.flags.DEFINE_boolean("train_embed", False,
#                            "Trains embeddings on data if True")

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

def word_embed_tokenizer(w1, w2, feature, embedding_index):
    """
    Preps the sentences to be embedded using Keras' tokenizer.
    """
    # vectorize the text samples into a 2D integer tensor
    #texts = w1 + w2 + feature
    texts = np.append(w1, [w2, feature])
    #texts = np.transpose(np.vstack((w1, w2, feature))).tolist()
    print("\ntexts shape = ", texts.shape)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    pad_seq = pad_sequences(sequences, maxlen=1)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Preparing embedding matrix.')

    num_words = len(word_index)
    # adding 1 because Tokenizer indices start at 1
    embedding_matrix = np.zeros((num_words+1, embedding_index["dog"].size))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words+1,
                                embedding_index["dog"].size,
                                weights=[embedding_matrix],
                                input_length=1,
                                trainable=FLAGS.train_embed)

    print('Computing embeddings.')

    input_shape = pad_seq.shape[1:]
    print("input_shape, should be 1 = ", input_shape)

    sequence_input = Input(shape=input_shape, dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Check the model if it embeds
    model = Model(sequence_input, embedded_sequences)
    # return the embedded input as np.arrays
    pad_seq = np.squeeze(np.asarray(pad_seq))
    print("\n shape pad_seq = ", pad_seq.shape)
    w1 = pad_seq[:len(w1)]
    w2 = pad_seq[len(w1):len(w1) + len(w2)]
    feature = pad_seq[-len(feature):]

    #data = np.vstack((w1, w2, feature))
    data = np.transpose(np.vstack((w1, w2, feature)))
    print("\n shape data = ", data.shape)

    return model, data
