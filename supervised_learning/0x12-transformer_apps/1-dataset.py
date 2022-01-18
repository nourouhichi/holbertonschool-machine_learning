#!/usr/bin/env python3
"""transformer archi"""
import tensorflow_datasets as tfds


class Dataset:
    """data preprocessing for transformer"""
    def __init__(self):
        """init function"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenizing data"""
        eng = []
        port = []
        for i, y in data:
            eng.append(y.numpy())
            port.append(i.numpy())
        token = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        return token(
                     eng, target_vocab_size=2**15), token(
                         port, target_vocab_size=2**15)

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        size_pt = self.tokenizer_pt.vocab_size
        size_eng = self.tokenizer_en.vocab_size
        return [size_pt] + self.tokenizer_pt.encode(
            pt.numpy()) + [size_pt + 1], [size_eng] + self.tokenizer_en.encode(
            en.numpy()) + [size_eng + 1]
