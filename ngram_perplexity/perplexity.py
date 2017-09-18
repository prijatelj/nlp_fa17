"""
Determine perplexity of English and French based on Google Books Ngrams
"""

import sys
from collections import Counter
import numpy as np
from google_ngram_downloader import readline_google_store as read_ngrams
from pathos.multiprocessing import ProcessingPool

def get_ngram_counter(ngram_len=1, lang="eng"):
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

def get_ngram_conditionals(ngrams, mgrams):
    """
    Finds the conditional probabilities that a ngram is the last gram in a
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

        if mgram not in mgrams:
            print("Something is wrong with the mgrams[mgram]")

        conditionals[ngram] = float(occurrences) / float(mgrams[mgram])

    return conditionals

def perplexity(lang="eng"):
    """
    finds satistical perplexity of the language model in Google Books N-Gram
    dataset.
    """
    pool = ProcessingPool(4)
    unigram_counter, mgram_counter, ngram_counter= pool.map(get_ngram_counter,
                                              [1,2,3],
                                              [lang] * 3)
    pool.close()
    pool.join()

    total_words = np.sum(np.array(list(unigram_counter.values())))
    print("total_words = ", total_words)

    ngram_conditionals = get_ngram_conditionals(ngram_counter,
                                            mgram_counter)

    probs = np.power(np.array(list(ngram_conditionals.values()),
                              dtype=np.float64),
             -np.array(list(ngram_counter.values()), dtype=np.float64) \
             / total_words)

    print("probs shape = ", probs.shape)

    PP = (np.prod(probs, dtype=np.float64))

    return PP

def main(args):
    print(perplexity(args[1]))
    #print(perplexity())

if __name__ == "__main__":
    main(sys.argv)
