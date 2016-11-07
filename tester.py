import re
import sys

from classifier import Classifier


def test_acc():
    cl = Classifier(sys.argv[1])
    with open(sys.argv[2]) as infile:
        for line in infile:
            # Dummy weights of 10000
            pred_classes = cl.classify(line.rstrip("\w"), [10000]*5, [10000]*5)
            print(line.rstrip("\w"))
            tagged_classes = infile.readline().rstrip("\n").split(" ")
            print(accuracy(pred_classes, tagged_classes))


def accuracy(pred_classes, tagged_classes):
    match = 0
    for k, v in zip(pred_classes, tagged_classes):
        if k == v:
            match += 2
        else:
            # We got at least 1 language right and weren't completely wrong
            if k == 'H/E':
                if v == 'H' or v == 'E':
                    match += 1
            elif v == 'H/E':
                if k == 'H' or k == "E":
                    match += 1
    return match * 100 / len(pred_classes) * 2
