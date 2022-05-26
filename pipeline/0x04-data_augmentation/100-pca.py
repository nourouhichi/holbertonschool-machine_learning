#!/usr/bin/env python3
"""pca module"""
import numpy as np


def pca_color(image, alphas):
    """ pca function"""
    x, y = image.shape[0], image.shape[1]
    img = np.reshape(image, (x * y, 3))
    mean = np.mean(img)
    std = np.std(img)
    img = img.astype('float32')
    img -= np.mean(img)
    img /= np.std(img)
    cov = np.cov(img, rowvar=False)
    lambdas, i = np.linalg.eig(cov)
    delta = np.dot(i, alphas * lambdas)
    pca_augmentation = img + delta
    pca = pca_augmentation * std + mean
    pca = pca.reshape(x, y, 3)
    pca = np.maximum(np.minimum(pca, 255), 0).astype(
                     'uint8')
    return pca
