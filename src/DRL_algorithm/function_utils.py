import sys
import numpy as np
import time

import tensorflow as tf

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"La fonction '{func.__name__}' a pris {execution_time:.9f} secondes pour s'exécuter.")
        return result
    return wrapper

def argmax(dict):
    top_value = float("-inf")
    ties = []
    for key in dict.keys():
        if dict[key] > top_value:
            ties.clear()
            ties.append(key)
            top_value = dict[key]
        elif dict[key] == top_value:
            ties.append(key)
    return np.random.choice(ties)
# @timing_decorator
def acceptable_softmax_with_mask(X: tf.Tensor, M: tf.Tensor):
    # positive_X = (X.T - np.min(X, axis=1)).T
    # masked_positive_X = positive_X * M
    # max_X = np.max(masked_positive_X, axis=1)
    # exp_X = np.exp(masked_positive_X.T - max_X).T
    # masked_exp_X = exp_X * M
    #
    # return (masked_exp_X.T / np.sum(masked_exp_X, axis=1)).T

    positive_X = tf.transpose(tf.transpose(X) - tf.reduce_min(X, axis=1))
    masked_positive_X = positive_X * M
    max_X = tf.reduce_max(masked_positive_X, axis=1)
    exp_X = tf.exp(tf.transpose(tf.transpose(masked_positive_X) - max_X))
    masked_exp_X = exp_X * M

    return tf.transpose(tf.transpose(masked_exp_X) / tf.reduce_sum(masked_exp_X, axis=1))

def apply_mask(X: np.ndarray, M: np.ndarray):
    X = X * M
    indices_zero = np.where(M == 0)
    X[indices_zero[0], indices_zero[1]] = -sys.float_info.max
    return X



