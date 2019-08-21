# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
import tensorflow as tf


def get_true_action(action):
    """
    用于将mcts选出的action转换成在unity中执行的action
    :param action:
    :return:
    """
    steer = -1.0 + (action // 11) * 0.2
    accelerator = -1.0 + (action % 11) * 0.2
    return [steer, accelerator,0,0,0,0,0]

def get_recoverOb(ob):
    """
    获得用于在Unity中还原状态的ob
    :param ob:
    :return:
    """
    return [0,0,1,ob[0],ob[1],ob[2],ob[3]];

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


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

# def fc(x, n_output, activation_fn=None):
#     """
#     全连接
#     :param x:
#     :param n_output:
#     :param activation_fn:
#     :return:
#     """
#     W=tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
#     b=tf.Variable(tf.constant(0.0, shape = [n_output]))
#     fc1 = tf.add(tf.matmul(x, W), b)
#     if not activation_fn == None:
#         fc1 = activation_fn(fc1)
#     return fc1


def flatten(x):
    """
    将4d tensor flatten成2d tensor
    :param x:
    :return:
    """
    return tf.reshape(x, [-1, int(x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3])])


