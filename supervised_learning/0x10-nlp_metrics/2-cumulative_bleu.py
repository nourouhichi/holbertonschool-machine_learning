#!/usr/bin/env python3
"""belu score calculation"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """ngram bleu score"""
    n_grams = []
    ref_grams = []
    occ = []
    num = 0
    sen = sentence.copy()
    for i in range(len(sen)):
        n_grams.append(sen[i:i + n])
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
    return num / len(n_grams)


def cumulative_bleu(references, sentence, n):
    """cum bleu score"""
    length = len(sentence)
    ref_len = np.array([len(i) for i in references])
    ref_min = np.argmin(np.abs(ref_len - length))
    i = len(references[ref_min])
    if i > length:
        pen = np.exp(1 - i / length)
    else:
        pen = 1
    grams = []
    for i in range(1, n + 1):
        grams.append((ngram_bleu(references, sentence, i)))
    grams = np.array(grams)
    sum = np.exp(np.sum((1 / n) * np.log(grams)))
    return pen * sum
