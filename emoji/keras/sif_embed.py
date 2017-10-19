"""
Very basic model that uses pretrained word vectors and the sif embedding as an
attempt to solve the STS task.
"""
from __future__ import print_function

import sys
import copy
import math
import numpy as np
from scipy.spatial.distance import cosine
import sklearn
from nltk import word_tokenize
import nltk
import tensorflow as tf

import sts_data_handler
#from summarizer_modules import str2vec_arora
from SIF_embedding import SIF_embedding as sif_embed

from ..data_handlers import sts_handler

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

def word_embed_np(sent_pairs, word_embed, token_dict):
    tokenized_pairs = copy.deepcopy(sent_pairs)
    for i in range(len(sent_pairs)):
        for j in range(len(sent_pairs[i])):
            sent_pairs[i, j] = np.asarray([word_embed[word] if word in
                word_embed.keys() else word_embed['']
                for word in sent_pairs[i, j]])
            tokenized_pairs[i, j] = np.asarray([token_dict[word] if word in
                token_dict.keys() else token_dict['']
                for word in sent_pairs[i, j]]).astype(int)

    return sent_pairs, tokenized_pairs

def sent_embed(sent_pairs, word_embed, fdist, first_pc, arora=True):
    embedded = []
    for sent_pair in sent_pairs:
        if arora:
            embedded.append([str2vec_arora(
                sent_pair[0], word_embed, fdist, first_pc),
                str2vec_arora(
                sent_pair[1], word_embed, fdist, first_pc)])
        else:
            embedded.append([str2vec_avg(sent_pair[0], word_embed, fdist),
                str2vec_avg(sent_pair[1], word_embed, fdist)])
    return np.asarray(embedded)

def findc(sents, word_embed, fdist):
    # Glove
    #matrix = np.stack([str2vec_avg(' '.join(sent)) for sent in sents], axis=1)
    matrix = np.stack([str2vec_avg(sent, word_embed, fdist) for sent in sents],
        axis=1)
    # Perform pca
    pca = sklearn.decomposition.PCA(n_components=5)
    c0 = pca.fit_transform(matrix)
    return c0

def normalizeVec(vec):
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec

def str2vec_avg(string, word_vectors, fdist):
    """
    Average vector for sentence using arora models. words = tokenized sentence
    """
    #words = word_tokenize(s)
    tot = np.zeros(300)
    for word in string:
        if word not in word_vectors:
            word = "" # default unknown token
        # TODO if not in word_vectors create random vector and do following:
        #Compute weighting according to smooth inverse frequencies
        a = .00001
        freq = fdist.freq(word)
        sif = a / (a + freq)
        normvec = normalizeVec(word_vectors[word])
        tot += (sif * normvec)
    return normalizeVec(tot)

def str2vec_arora(string, word_embed, fdist, c0):
    r = str2vec_avg(string, word_embed, fdist)
    vec = c0[:, 0]
    vec = np.reshape(vec, np.shape(r))
    r -= (np.inner(r, vec)/np.inner(vec, vec))*vec
    return normalizeVec(r)

def avg_arora(sent_pairs, word_embed, arora=True):
    """
    avg of word embeds = sent embed. 0 or 1 for no removal or to remove the
    principal component.
    """
    fdist = nltk.FreqDist(nltk.corpus.brown.words())

    sents = np.append(sent_pairs[:,0], sent_pairs[:,1])
    print(sents.shape)
    c0 = findc(sents, word_embed, fdist)

    if arora:
        embed_arora = sent_embed(sent_pairs, word_embed, fdist, c0)
    else:
        embed_arora = sent_embed(sent_pairs, word_embed, fdist, c0, False)

    print("embed_arora shape = ", embed_arora.shape)

    #embed_pair, token_pair = word_embed_np(
    #    sent_pairs, word_embed, word_to_int)

    # TODO finish implementing train_valid_split
    #train, valid = sts_data_handler.train_valid_split(
    #    [rates, embed_pair, token_pair])

    #print(type(embed[0]))

    #embed1 = sif_embed(embed, token_pair[:, 0], embed_pair[:, 0], 0)
    #embed2 = sif_embed(embed, token_pair[:, 1], embed_pair[:, 1], 1)


    return embed_arora

def embed(test=False):
    #rates, sent_pairs = sts_data_handler.read_tsv()
    sent_pairs, rates = sts_data_handler.read_tsv()

    vocab, embed = sts_handler.load_pretrained_glove()
    word_embed = dict(zip(vocab, embed))

    print("sent_pairs.shape = ", sent_pairs.shape)
    print("rates.shape = ", rates.shape)

    embed_arora = avg_arora(sent_pairs, word_embed)
    embed_avg = avg_arora(sent_pairs, word_embed, False)

    if test:
        test_pairs, test_rates = sts_data_handler.read_tsv(FLAGS.sts_test_tsv)
        test_embed_arora = avg_arora(test_pairs, word_embed)
        test_embed_avg = avg_arora(test_pairs, word_embed, False)


    """ NaN's exist for some reason in prediction
    #check cosine distance
    prediction = []
    count = 0
    infc = 0
    for i, pair in enumerate(embed_arora):
        if np.array_equal(pair[0], np.zeros(pair[0].shape)):
            print(sent_pairs[i, 0])
            print(embed_arora[i, 0])
        if np.array_equal(pair[1], np.zeros(pair[1].shape)):
            print(sent_pairs[i, 1])
            print(embed_arora[i, 1])
        prediction.append((cosine(pair[0], pair[1]) + 1) * (5 / 2))

        #if math.isnan(prediction[-1]):
        #    print(prediction[-1])
        #    count += 1
        #    print(pair[0])
        #    print(pair[1])

    #print("count o NaNs = ", count)
    prediction = np.asarray(prediction)

    pred_avg = []
    for pair in embed_avg:
        pred_avg.append((cosine(pair[0], pair[1]) + 1) * (5 / 2))
    pred_avg = np.asarray(pred_avg)

    print(prediction.shape)
    print(type(prediction[0]), " ", prediction[0])
    print(type(rates[0]), " ", rates[0])

    print(np.array_equal(np.isnan(rates), np.zeros(rates.shape, dtype=bool)))
    print(np.array_equal(np.isnan(prediction),
          np.zeros(prediction.shape, dtype=bool)))

    print(np.array_equal(np.isfinite(rates), np.ones(rates.shape, dtype=bool)))
    print(np.array_equal(np.isfinite(prediction),
          np.ones(prediction.shape, dtype=bool)))

    mse = sklearn.metrics.mean_squared_error(rates, prediction)
    mse_avg = sklearn.metrics.mean_squared_error(rates, pred_avg)


    #print("MSE = (SUM for n examples: (Y - y)^2) / n")
    print("In-Sample Accuracy:")
    print("mse of arora: ", mse)
    print("Accuracy = (1 - mse/MAX_ERROR) = ", 1 - (mse/25))
    print("mse of avg: ", mse_avg)
    print("Accuracy = (1 - mse/MAX_ERROR) = ", 1 - (mse_avg/25))
    #"""
    if test:
        np.save("data/embed_test_train_arora", embed_arora)
        np.save("data/embed_test_train_avg", embed_avg)
        np.save("data/test_train_rates", rates)

        np.save("data/embed_test_arora", test_embed_arora)
        np.save("data/embed_test_avg", test_embed_avg)
        np.save("data/test_rates", test_rates)

    else:
        np.save("data/embed_arora", embed_arora)
        np.save("data/embed_avg", embed_avg)
        np.save("data/rates", rates)

def main(argv):
    embed()

if __name__ == "__main__":
    tf.app.run()
