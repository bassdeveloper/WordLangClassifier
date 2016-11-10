import sys

from classifier import Classifier


def test_acc():
    cl = Classifier(sys.argv[1])
    avg_acc = 0
    count = 0
    with open(sys.argv[2]) as infile:
        for line in infile:
            pred_classes = cl.classify(line.rstrip("\w"))
            tagged_classes = infile.readline().rstrip("\n").split(" ")
            print(line.lower().rstrip("\w").rstrip("\n"))
            acc = accuracy(pred_classes, tagged_classes)
            print(pred_classes)
            print(tagged_classes)
            print(acc)
            avg_acc += acc * (len(pred_classes)) / 100
            count += len(pred_classes)

    print('Average accuracy: ', avg_acc * 100 / count)


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
    return match * 100 / (len(pred_classes) * 2)


test_acc()
