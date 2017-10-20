'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

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

def word_embed_tokenizer(sents, embedding_index):
    """
    Preps the sentences to be embedded using Keras' tokenizer.
    """
    texts = sents.tolist()

    # calculate maximum num of words in a sentence
    max_sequence_length = 0
    for s in texts:
        max_sequence_length = max(max_sequence_length, len(s.split()))

    # vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    pad_seq = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # TODO THIS IS FOR multiple class classification! NOT regressions!
    #labels = to_categorical(np.asarray(labels))
    print('Shape of pad_seq tensor:', pad_seq.shape)
    #print('Shape of label tensor:', labels.shape)

    print('Preparing embedding matrix.')

    num_words = len(word_index)
    # adding 1 because Tokenizer indices start at 1
    embedding_matrix = np.zeros((num_words+1, embedding_index["the"].size))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words+1,
                                embedding_index["the"].size,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=FLAGS.train_embed)

    print('Computing embeddings.')

    input_shape = pad_seq.shape[1:]

    sequence_input = Input(shape=input_shape, dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Check the model if it embeds
    model = Model(sequence_input, embedded_sequences)
    return model, pad_seq
    #return model, pad_seq, labels
