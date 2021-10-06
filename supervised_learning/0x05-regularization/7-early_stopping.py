#!/usr/bin/env python3
""" Regularization module"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Dearly stop gradient descent"""
    if opt_cost <= cost + threshold:
        count += 1
    else:
        count = 0
    if count >= patience:
        return True, count
    return False, count
