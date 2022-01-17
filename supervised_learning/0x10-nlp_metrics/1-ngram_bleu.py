#!/usr/bin/env python3
"""belu score calculation"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """ngram bleu score"""
    n_grams = []
    ref_grams = []
    occ = []
    num = 0
    length = len(sentence)
    ref_len = np.array([len(i) for i in references])
    ref_min = np.argmin(np.abs(ref_len - length))
    i = len(references[ref_min])
    if i > length:
        pen = np.exp(1 - i / length)
    else:
        pen = 1
    for i in range(len(sentence)):
        n_grams.append(sentence[i:i + n])
    if len(n_grams[-1]) < n:
        n_grams.pop(-1)
    for y in references:
        ref = []
        for le in range(len(y)):
            ref.append(y[le:le+n])
        ref_grams.append(ref)
    for i in n_grams:
        for y in ref_grams:
            occ.append(y.count(i))
        num += max(occ)
        occ = []
    return pen * num / len(n_grams)
