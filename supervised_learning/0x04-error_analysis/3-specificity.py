#!/usr/bin/env python3
"""new module"""

import numpy as np


def specificity(confusion):
    """calculates specificity of a confusion matrix"""
    tp = confusion.diagonal()
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    tn = confusion.sum() - (tp + fp + fn)
    return tn / (tn + fp)
