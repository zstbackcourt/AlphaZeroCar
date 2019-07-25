# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import utils
import numpy as np
import tensorflow as tf


class dqn(object):
    def __init__(self,epsilon,epsilon_anneal,end_epsilon,lr,gamma,state_size,action_size,name_scope):
        """

        :param sess:
        :param epsilon: e-greedy探索的系数
        :param epsilon_anneal: epsilon的线性衰减率
        :param end_epsilon: 最低的探索比例
        :param lr: learning rate
        :param gamma: 折扣率
        :param state_size: observation dim
        :param action_size: action dim
        :param name_scope: 命名域
        """
        self.sess = tf.Session()
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.lr = lr
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.name_scope = name_scope
        self.qnetwork()
        self.sess.run(tf.global_variables_initializer())
    def qnetwork(self):
        """
        创建Q network
        注：Value就是所有Q的和 v = sum(π*q)
        :return:
        """
        with tf.variable_scope(self.name_scope):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size])  # 状态输入
            self.action = tf.placeholder(tf.int32, [None])  # 动作输入
            self.target_q = tf.placeholder(tf.float32, [None])  # target Q

            fc1 = utils.fc(self.state_input, n_output=16, activation_fn=tf.nn.relu)
            fc2 = utils.fc(fc1, n_output=32, activation_fn=tf.nn.relu)
            fc3 = utils.fc(fc2, n_output=16, activation_fn=tf.nn.relu)
            # self.vf = utils.fc(fc3,1,activation_fn=tf.nn.relu)
            self.q_values = utils.fc(fc3, self.action_size, activation_fn=None) # 每一个动作的q_value
            self.pi = utils.fc(fc3,self.action_size,activation_fn=tf.nn.log_softmax) # 动作分布
            # self.vf = tf.reduce_sum(tf.multiply(self.q_values,self.pi),1)
            # 动作用one-hot编码
            action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0) # 将输入的action编码成one-hot
            # 预测的q
            self.q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)
            # self.q_value_pred = utils.fc(fc3,n_output=1,activation_fn=tf.nn.tanh)
            # q network的loss
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, self.q_value_pred)))
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss)


    # def get_action_values(self, state):
    #     actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
    #     return actions
    #
    # def get_optimal_action(self, state):
    #     """
    #     最优action就是对应的最大value的action
    #     :param state:
    #     :return:
    #     """
    #     actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
    #     return actions.argmax()

    # def get_action(self, state):
    #     """
    #     用e-greedy策略选择与环境交互的action
    #     :param state:
    #     :return:
    #     """
    #     if np.random.random() < self.epsilon:
    #         # 以epsilon的概率随机选择一个动作
    #         return np.random.randint(0, self.action_size)
    #     else:
    #         return self.get_optimal_action(state)

    # def epsilon_decay(self):
    #     """
    #     epsilon衰减
    #     :return:
    #     """
    #     if self.epsilon > self.end_epsilon:
    #         self.epsilon -= self.epsilon_anneal

    def learn(self, buffer, num_steps, batch_size):
        if buffer.size() <= batch_size:
            print("buffer size:",buffer.size())
            return
        else:
            for step in range(num_steps):
                print("第{}次更新".format(step))
                # states,next_states,actions,rewards,dones,mcts_probs,values
                minibatch = buffer.get_batch(batch_size=batch_size)
                state_batch = [data[0] for data in minibatch]
                next_state_batch = [data[1] for data in minibatch]
                action_batch = [data[2] for data in minibatch]
                reward_batch = [data[3] for data in minibatch]
                done_batch = [data[4] for data in minibatch]
                mcts_prob_batch = [data[5] for data in minibatch]
                value_batch = [data[6] for data in minibatch]
                # state_batch = [data[0] for data in minibatch]
                # action_batch = [data[1] for data in minibatch]
                # reward_batch = [data[2] for data in minibatch]
                # next_state_batch = [data[3] for data in minibatch]
                # done_batch = [data[4] for data in minibatch]
                state_batch = np.array(state_batch).reshape((-1,self.state_size))
                next_state_batch = np.array(next_state_batch).reshape((-1,self.state_size))
                action_batch = np.array(action_batch).reshape((batch_size))
                # q_values = self.sess.run(self.q_values, feed_dict={self.state_input: next_state_batch})
                # max_q_values = q_values.max(axis=1)
                q_values = self.sess.run(self.q_values, feed_dict={self.state_input: next_state_batch})
                # 计算target q value
                # target_q = np.array(
                #     [data[2] + self.gamma * max_q_values[i] * (1 - data[4]) for i, data in enumerate(minibatch)]
                # )
                max_q_values = q_values.max(axis=1)
                target_q = np.array(
                    [data[3] + self.gamma * max_q_values[i] * (1 - data[4]) for i, data in enumerate(minibatch)]
                )
                # print(target_q)
                target_q = target_q.reshape([batch_size])
                #print(target_q,target_q.shape)
                # 最小化TD-error,即训练
                l, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.state_input: state_batch,
                    self.target_q: target_q,
                    self.action: action_batch
                })


    def policy_value(self,state):
        """
        输入:a batch of states
        :param state:
        :return: a batch of action probabilities and state values
        """
        # log_act_probs,value = self.sess.run([self.pi,self.q_value_pred],feed_dict={self.state_input: [state]})
        # act_probs = np.exp(log_act_probs)
        log_act_probs, q = self.sess.run([self.pi, self.q_values], feed_dict={self.state_input: [state]})
        act_probs = np.exp(log_act_probs)
        value = np.sum(np.multiply(act_probs,q),axis=1)
        # '''测试代码'''
        # log_act_probs,q_values,vf = self.sess.run([self.q_values,self.pi,self.vf],feed_dict={self.state_input: [state]})
        # print("测试:",log_act_probs,q_values,vf)

        return act_probs, value

    def policy_value_fn(self, action, state):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        act_probs, value = self.policy_value(state[0])
        # print(act_probs, value)
        return zip(action, act_probs[0]), value[0]