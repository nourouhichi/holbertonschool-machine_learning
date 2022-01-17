#!/usr/bin/env python3
"""belu score calculation"""
import numpy as np


def uni_bleu(references, sentence):
    """unigram bleu score"""
    num = 0
    occ = []
    length = len(sentence)
    ref_len = np.array([len(i) for i in references])
    ref_min = np.argmin(np.abs(ref_len - length))
    i = len(references[ref_min])
    if i > length:
        pen = np.exp(1 - i / length)
    else:
        pen = 1
    new_sentence = list(dict.fromkeys(sentence))
    for i in new_sentence:
        for y in references:
            occ.append(y.count(i))
        num += max(occ)
        occ = []
    return pen * num / length
