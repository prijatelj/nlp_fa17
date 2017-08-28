"""
Counts the top 50 words in the given English
Python 3.3+

@author: Derek S. Prijatelj
"""

import sys
import regex
from collections import Counter

def top_kwords_ws_split(filepath, k):
    """
    Finds top k most frequent words and total word count in given .txt file.
    Words are any non-ws char sequence divided by ws.
    """
    with open(filepath, 'r', encoding="utf8") as txt:
        word_count = Counter(txt.read().casefold().split())

    return word_count.most_common(k), len(list(word_count.elements()))

def top_kwords_alphanum(filepath, k):
    """
    Finds top k most frequent words and total word count in given .txt file.
    Words are defined by the provided regex word_defined.
    """
    with open(filepath, 'r', encoding="utf8") as txt:
        txt_content = txt.read().casefold()
        words = regex.findall(r"[\p{alpha}\p{digit}]+", txt_content)
        #generalized utf8 encoded regex = multi-lingual for at least Latin text
        word_count = Counter(words)

    return word_count.most_common(k), len(list(word_count.elements()))

def print_out(words, total):
    #Sorts ties lexically and overall by decsending token occurrences
    words.sort(key=lambda x: x[0])
    words.sort(key=lambda x: x[1], reverse=True)

    print("word\tcount\t% frequency")
    print("----------------------")
    for word in words:
        print("{0}\t{1:d}\t{2:.2f}%".format(word[0], word[1],
              float(word[1])/float(total)))

def main(argv):
    # argv = python_file.py filepath k [en | fr]
    print(argv[1])
    if len(argv) == 3:
        # default = filepath k (w/ white space split)
        print_out(*top_kwords_ws_split(argv[1], int(argv[2])))
    elif len(argv) == 4:
        print_out(*top_kwords_alphanum(argv[1], int(argv[2])))

if __name__ == "__main__":
    main(sys.argv)
