"""
Simple script that always guesses the most common emoji as its prediction. This serves as the baseline to surpass for class. Saves output into file.
"""
import sys

def guess(test_labels, fp):
    with open(test_labels) as labels, open(fp, 'w') as output:
        content = labels.readlines()
        for line in content:
            output.write("0\n")

def main(args):
    guess(args[1], args[2])

if __name__ == "__main__":
    main(sys.argv)
