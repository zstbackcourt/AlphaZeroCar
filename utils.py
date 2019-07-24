# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
import tensorflow as tf


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def lkrelu(x, slope=0.05):
    return tf.maximum(slope * x, x)


def max_pool(x, k_sz=[2,2]):
    """
    最大池化
    :param x:
    :param k_sz:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, k_sz[0], k_sz[1], 1], strides=[1, k_sz[0], k_sz[1], 1], padding='SAME')

def conv2d(x, n_kernel, k_sz, stride=1):
    """
    2d卷积
    :param x:
    :param n_kernel:
    :param k_sz:
    :param stride:
    :return:
    """
    W = tf.Variable(tf.random_normal([k_sz[0], k_sz[1], int(x.get_shape()[3]), n_kernel]))
    b = tf.Variable(tf.random_normal([n_kernel]))
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b) # add bias term
    return tf.nn.relu(conv) # rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)


def fc(x, n_output, activation_fn=None):
    """
    全连接
    :param x:
    :param n_output:
    :param activation_fn:
    :return:
    """
    W=tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
    b=tf.Variable(tf.random_normal([n_output]))
    fc1 = tf.add(tf.matmul(x, W), b)
    if not activation_fn == None:
        fc1 = activation_fn(fc1)
    return fc1


def flatten(x):
    """
    将4d tensor flatten成2d tensor
    :param x:
    :return:
    """
    return tf.reshape(x, [-1, int(x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3])])


