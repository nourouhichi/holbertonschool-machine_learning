  
#!/usr/bin/env python3
"""clustering"""
import numpy as np


def initialize(X, k):
    """random centroids"""
    if type(k) != int or k <= 0:
        return None
    try:
        maxi = np.amax(X, axis=0)
        mini = np.amin(X, axis=0)
        return np.random.uniform(mini, maxi, (k, X.shape[1]))
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """iteration operation"""
    if type(iterations) != int or iterations <= 0:
        return None, None
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    try:
        n, d = X.shape
        clss = np.ndarray((n,))
        C = np.ndarray((k, d))
        for _ in range(iterations):
            dist = np.sum(np.square(X[
                :, np.newaxis, :] - centroids), axis=2)
            clss = np.argmin(dist, axis=1)
            for x in range(k):
                if x not in clss:
                    maxi = np.max(X, axis=0)
                    mini = np.min(X, axis=0)
                    C[x] = np.random.uniform(mini, maxi)
                else:
                    C[x] = np.mean(X[np.argwhere(
                        clss == x).reshape(-1)], axis=0)
            if(C == centroids).all():
                return centroids, clss
            centroids = C.copy()
        return C, clss
    except Exception:
        return None, None  
#!/usr/bin/env python3
"""clustering"""
import numpy as np


def initialize(X, k):
    """random centroids"""
    if type(k) != int or k <= 0:
        return None
    try:
        maxi = np.amax(X, axis=0)
        mini = np.amin(X, axis=0)
        return np.random.uniform(mini, maxi, (k, X.shape[1]))
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """iteration operation"""
    if type(iterations) != int or iterations <= 0:
        return None, None
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    try:
        n, d = X.shape
        clss = np.ndarray((n,))
        C = np.ndarray((k, d))
        for _ in range(iterations):
            dist = np.sum(np.square(X[
                :, np.newaxis, :] - centroids), axis=2)
            clss = np.argmin(dist, axis=1)
            for x in range(k):
                if x not in clss:
                    maxi = np.max(X, axis=0)
                    mini = np.min(X, axis=0)
                    C[x] = np.random.uniform(mini, maxi)
                else:
                    C[x] = np.mean(X[np.argwhere(
                        clss == x).reshape(-1)], axis=0)
            if(C == centroids).all():
                return centroids, clss
            centroids = C.copy()
        return C, clss
    except Exception:
        return None, None
