"""
Counts the top 50 words in the given English
Python 3.3+

@author: Derek S. Prijatelj
"""

import sys
import regex
from collections import Counter

def top_kwords(filepath, k=50, word_defined=""):
    """
    records the top k words in a text document in either english or french.
    """
    if word_defined == "":
        return top_kwords_ws_split(filepath, k)
    else:
        return top_kwords_word_defined(filepath, k, word_defined)

def top_kwords_ws_split(filepath, k):
    """
    Finds top k most frequent words and total word count in given .txt file.
    Words are any non-ws char sequence divided by ws.
    """
    with open(filepath, 'r', encoding="utf8") as txt:
        word_count = Counter(txt.read().casefold().split())

    return word_count.most_common(k), len(list(word_count.elements()))

def top_kwords_word_defined(filepath, k, word_defined):
    """
    Finds top k most frequent words and total word count in given .txt file.
    Words are defined by the provided regex word_defined.
    """
    with open(filepath, 'r', encoding="utf8") as txt:
        txt_content = txt.read().casefold()
        #words = regex.findall(word_defined, txt_content)
        words = regex.findall(r"[\p{alpha}\p{digit}]+", txt_content)
        word_count = Counter(words)

    return word_count.most_common(k), len(list(word_count.elements()))

def print_out(words, total):
    #Sorts ties lexically and overall by decsending token freq
    words.sort(key=lambda x: x[0])
    words.sort(key=lambda x: x[1], reverse=True)

    for word in words:
        print("{0}\t{1:d}\t{2:.2f}".format(word[0], word[1],
              float(word[1])/float(total)))

def main(argv):
    # argv = python_file.py filepath k [en | fr]
    if len(argv) == 3:
        # default = filepath k (w/ white space split)
        print_out(*top_kwords(argv[1], int(argv[2])))
    elif len(argv) == 4:
        print_out(*top_kwords(
            argv[1],
            int(argv[2]),
            r"[\p{alpha}\p{digit}]+", #generalized in code = multi-lingual
            ))

if __name__ == "__main__":
    main(sys.argv)
