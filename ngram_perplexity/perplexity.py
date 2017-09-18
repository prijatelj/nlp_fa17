"""
Determine perplexity of English and French based on Google Books Ngrams
"""

import sys
from collections import Counter
import numpy as np
from google_ngram_downloader import readline_google_store as read_ngrams
from pathos.multiprocessing import ProcessingPool

def get_n_gram_counter(ngram_len=1, lang="eng"):
    _, _, records = next(read_ngrams(ngram_len=ngram_len, lang=lang))
    ngram_counter = Counter()

    try:
        while True:
            record = next(records)

            # totals all ngram occurrences across years
            ngram_counter[record.ngram] += record.match_count

    except StopIteration:
        pass

    return ngram_counter

def ngram_conditionals(ngrams, mgrams):
    """
    Finds the conditional probabilities that a unigram is the last gram in a
    ngram.

    @param ngrams: Counter of all occurrences of ngrams in corpus
    @param mgrams: Counter of all occurrences of n-1 grams in corpus

    @returns: Dictionary with keys as ngrams and values as the probability
        that the ngram occurs out of all occurrences of the n-1_gram prefix
        in all ngrams.
    """
    conditionals = {}

    for ngram, occurrences in ngrams.items():
        grams = ngram.split(" ")

        if len(grams) > 1:
            mgram = " ".join(grams[:-1])
        else:
            mgram = grams[0]

        conditionals[ngram] = float(occurrences) / float(mgrams[mgram])

    return conditionals

def perplexity(lang="eng"):
    """
    finds satistical perplexity of the language model in Google Books N-Gram
    dataset.
    """
    pool = ProcessingPool(4)
    unigram_counter, bigram_counter= pool.map(get_n_gram_counter, [1, 2])
    pool.close()
    pool.join()

    #unigram_counter = get_n_gram_counter(ngram_len=1, lang=lang)
    #bigram_counter = get_n_gram_counter(ngram_len=2, lang=lang)

    bigram_conditionals = ngram_conditionals(bigram_counter,
                                             unigram_counter)

    probs = np.array(list(bigram_conditionals.values()), dtype=np.float) ** -1
    PP = (np.prod(probs)) ** -len(probs)

    return PP

def main(args):
    #perplexity(args[0])
    print(perplexity())

if __name__ == "__main__":
    main(sys.argv)
