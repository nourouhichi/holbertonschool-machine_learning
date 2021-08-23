#!/usr/bin/env python3
"""new module"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for x in range(m):
        for y in range(classes):
            if labels[x, y] == 1:
                ind1 = y
            if logits[x, y] == 1:
                ind2 = y
        confusion[ind1, ind2] += 1
    return confusion
