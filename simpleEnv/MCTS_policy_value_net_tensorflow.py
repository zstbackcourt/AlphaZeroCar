# -*- coding: utf-8 -*-

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
import numpy as np
import tensorflow as tf
from gym import spaces
from baselines.common import explained_variance, set_global_seeds


def lkrelu(x, slope=0.05):
    return tf.maximum(slope * x, x)


class PolicyValueNet(object):
    def __init__(self,
                 ob_space,
                 ac_space,
                 nbatch,
                 save_path,
                 reuse=False,
                 policy_coef=1.0,
                 value_coef=1.0,
                 l2_coef=1.0):  # pylint: disable=W0613

        set_global_seeds(0)

        if isinstance(ac_space, spaces.Box):
            act_dim = ac_space.shape[0]
        elif isinstance(ac_space, spaces.Discrete):
            act_dim = ac_space.n
        else:
            raise NotImplementedError

        if isinstance(ob_space, spaces.Box):
            ob_dim = ob_space.shape[0]
        elif isinstance(ob_space, spaces.Discrete):
            ob_dim = ob_space.n
        else:
            raise NotImplementedError

        X = tf.placeholder(tf.float32, [None, ob_dim], name='Ob')  # obs

        with tf.variable_scope("policyAndValue", reuse=reuse):

            p1 = lkrelu(fc(X, 'pi_fc1', nh=16, init_scale=np.sqrt(2.0)))
            p2 = lkrelu(fc(p1, 'pi_fc2', nh=32, init_scale=np.sqrt(2.0)))
            p3 = lkrelu(fc(p2, 'pi_fc3', nh=16, init_scale=np.sqrt(2.0)))

            v1 = lkrelu(fc(X, 'vf_fc1', nh=16, init_scale=np.sqrt(2.0)))
            v2 = lkrelu(fc(v1, 'vf_fc2', nh=32, init_scale=np.sqrt(2.0)))
            v3 = lkrelu(fc(v2, 'vf_fc3', nh=16, init_scale=np.sqrt(2.0)))

            pi = tf.nn.log_softmax(fc(p3, 'pi', nh=act_dim, init_scale=np.sqrt(2.0)))

            vf = fc(v3, 'vf', nh=1, init_scale=np.sqrt(2.0))

        self.X = X
        self.pi = pi
        self.vf = vf
        self.save_path = save_path

        """价值网络损失函数"""
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.value_loss = tf.losses.mean_squared_error(self.labels, self.vf)

        """策略网络损失函数"""
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, act_dim])

        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.pi), 1)))

        """L2正则化"""
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        """损失函数"""
        self.loss = value_coef * self.value_loss + policy_coef * self.policy_loss + l2_coef * l2_penalty

        """定义训练的优化器"""
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        """计算策略熵，仅用于监视"""
        self.entropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.pi) * self.pi, 1)))

        init = tf.global_variables_initializer()
        self.session.run(init)

        '''load saved model'''
        self.saver = tf.train.Saver(max_to_keep=1)
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find saved model")

    def policy_value(self, ob):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run([self.pi, self.vf], feed_dict={self.X: ob})
        act_probs = np.exp(log_act_probs)

        # print(log_act_probs)
        # print(act_probs)
        # print(act_probs[0].index(max(act_probs[0])))
        return act_probs, value

    def policy_value_fn(self, action, ob):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        act_probs, value = self.policy_value(ob)

        return zip(action, act_probs[0]), value

    def train_step(self, state_batch, mcts_probs, value_batch, lr):
        """perform a training step"""
        loss, entropy, _ = self.session.run([self.loss, self.entropy, self.optimizer],
                                            feed_dict={self.X: state_batch,
                                                       self.mcts_probs: mcts_probs,
                                                       self.labels: value_batch,
                                                       self.learning_rate: lr})
        return loss, entropy

    def save_model(self):
        print("保存模型")
        self.saver.save(self.session, self.save_path)
