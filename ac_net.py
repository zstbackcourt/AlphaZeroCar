# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import tensorflow as tf
import numpy as np

GLOBAL_NET_SCOPE = 'Global_Net'

class ACNet(object):
    def __init__(self, sess,N_S,N_A,scope,lra,lrc,globalAC=None,entropy_beat = 0.001):
        self.sess = sess
        self.N_S = N_S
        self.N_A = N_A
        self.optimizerA = tf.train.RMSPropOptimizer(lra, name='RMSPropA')
        self.optimizerC = tf.train.RMSPropOptimizer(lrc, name='RMSPropC')
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
                self.saver = tf.train.Saver(max_to_keep=1)
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, self.N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = entropy_beat * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.optimizerA.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.optimizerC.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a_1 = tf.layers.dense(self.s, 128, tf.nn.relu6, kernel_initializer=w_init, name='la1')
            l_a_2 = tf.layers.dense(l_a_1,256,tf.nn.relu6,kernel_initializer=w_init, name='la2')
            a_prob = tf.layers.dense(l_a_2, self.N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local\
        s = np.array(s).reshape((-1,self.N_S))
        # print(s.shape)
        # prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
