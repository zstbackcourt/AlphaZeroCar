# -*- coding:utf-8 -*-
"""
策略价值网络
@author: Weijie Shen
"""
from baselines.a2c.utils import fc
import numpy as np
import tensorflow as tf
from gym import spaces
from baselines.common import set_global_seeds
from mcts.logger import MyLogger
# from logger import MyLogger
#import ray
#import ray.experimental.tf_utils

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
        self.mylogger = MyLogger("./MCTSlog/logs/")
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

        X = tf.placeholder(tf.float32, [None, ob_dim-7], name='Ob')  # obs

        with tf.variable_scope("policyAndValue", reuse=reuse):

            p1 = lkrelu(fc(X, 'pi_fc1', nh=256, init_scale=np.sqrt(2.0)))
            p2 = lkrelu(fc(p1, 'pi_fc2', nh=128, init_scale=np.sqrt(2.0)))
            p3 = lkrelu(fc(p2, 'pi_fc3', nh=64, init_scale=np.sqrt(2.0)))
            # 油门
            # print(act_dim)
            pi = tf.nn.log_softmax(fc(p3, 'pi', nh=act_dim, init_scale=np.sqrt(2.0)))
            # 刹车
            # pi_2 = tf.nn.log_softmax(fc(p3, 'pi_2', nh=act_dim, init_scale=np.sqrt(2.0)))

            vf = fc(p3, 'vf', nh=1, init_scale=np.sqrt(2.0))

        self.X = X
        self.pi = pi
        # self.pi_2 = pi_2
        self.vf = vf
        self.save_path = save_path

        # 定义全局要执行的步数
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        """价值网络损失函数"""
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.value_loss = tf.losses.mean_squared_error(self.labels, self.vf)

        """策略网络损失函数"""
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, act_dim])

        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.pi), 1)))

        """L2正则化"""
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        self.l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        """损失函数"""
        self.loss = value_coef * self.value_loss + policy_coef * self.policy_loss + l2_coef * self.l2_penalty

        """定义训练的优化器"""
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)

        # Make a session
        self.session = tf.Session()
        self.mylogger.add_sess_graph(self.session.graph)
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
        # self.variables = ray.experimental.tf_utils.TensorFlowVariables(self.loss, self.session)
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

        return zip(action, act_probs[0]), value[0][0]

    def train_step(self, state_batch, mcts_probs, value_batch, lr):
        """perform a training step"""
        _loss, _value_loss, _policy_loss, _l2_penalty, _entropy, _ = self.session.run(
            [self.loss, self.value_loss, self.policy_loss, self.l2_penalty, self.entropy, self.optimizer],
            feed_dict={self.X: state_batch,
                       self.mcts_probs: mcts_probs,
                       self.labels: value_batch,
                       self.learning_rate: lr})
        # return _loss, _value_loss, _policy_loss, _l2_penalty, _entropy
        global_step = self.session.run(self.global_step)
        self.mylogger.write_summary_scalar(global_step, "loss", _loss)
        self.mylogger.write_summary_scalar(global_step, "value_loss", _value_loss)
        self.mylogger.write_summary_scalar(global_step, "policy loss",_policy_loss)
        self.mylogger.write_summary_scalar(global_step, "l2 penalty", _l2_penalty)
        self.mylogger.write_summary_scalar(global_step, "entropy", _entropy)
        return _loss, _value_loss, _policy_loss, _l2_penalty, _entropy

    # def get_weights(self):
    #     return self.variables.get_weights()

    # def set_weights(self,weights):
    #     self.variables.set_weights(weights)

    def save_model(self):
        print("保存模型")
        self.saver.save(self.session, self.save_path,global_step=self.global_step)