"""
splits the provided text file and corresponding label file into train, dev, and
test portions.
"""
import sys

def partition(original_text, original_labels,
          ftrain_text="ftrain_text.txt", ftrain_labels="ftrain_labels.txt",
          fdev_text="fdev_text.txt", fdev_labels="fdev_labels.txt",
          ftest_text="ftest_text.txt", ftest_labels="ftest_labels.txt"
         ):
    with open(original_labels) as labels, open(original_text) as text, \
             open(ftrain_text,'w',encoding="utf-8") as train_text, \
             open(ftrain_labels,'w',encoding="utf-8") as train_labels, \
             open(fdev_text,'w',encoding="utf-8") as dev_text, \
             open(fdev_labels,'w',encoding="utf-8") as dev_labels, \
             open(ftest_text,'w',encoding="utf-8") as test_text, \
             open(ftest_labels,'w',encoding="utf-8") as test_labels, \
             open("baseline_dev.txt" , 'w',encoding="utf-8") as baseline_dev, \
             open("baseline_test.txt", 'w',encoding="utf-8") as baseline_test:
        label_content = labels.readlines()
        text_content = text.readlines()
        assert len(label_content) == len(text_content)

        total = len(label_content)
        train_part = total * .5
        dev_part = total * .25

        for i, line in enumerate(text_content):
            if i < train_part:
                train_text.write(line)
                train_labels.write(label_content[i])
            elif i < train_part + dev_part:
                dev_text.write(line)
                dev_labels.write(label_content[i])
                baseline_dev.write("0\n")
            else:
                test_text.write(line)
                test_labels.write(label_content[i])
                baseline_test.write("0\n")

def main(args):
    partition(args[1], args[2])

if __name__ == "__main__":
    main(sys.argv)
