# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import utils
import numpy as np
import tensorflow as tf
from logger import MyLogger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class dqn(object):
    def __init__(self,epsilon,epsilon_anneal,end_epsilon,lr,gamma,state_size,action_size,name_scope,save_path):
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
        :param save_path: 模型保存的路径
        """
        self.sess = tf.Session()
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.lr = lr
        self.lr_multiplier = 1.0
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.name_scope = name_scope
        self.save_path = save_path
        self.policy_coef = 1
        self.value_coef = 1
        self.mylogger = MyLogger("./logs/")
        # 定义全局要执行的步数
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=3)
        self.qnetwork()
        self.loadModel()
        self.sess.run(tf.global_variables_initializer())




    def loadModel(self):
        """
        如果存在pretrained模型就加载
        :return:
        """
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find saved model!")

    def saveModel(self):
        self.saver.save(self.sess,self.save_path,global_step=self.global_step)
        print("save the latest model sucessfully!")

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
            self.mcts_probs = tf.placeholder(tf.float32, shape=[None, self.action_size])
            self.learning_rate = tf.placeholder(tf.float32)
            # fc1 = utils.fc(self.state_input, n_output=32, activation_fn=tf.nn.relu)
            # fc2 = utils.fc(fc1, n_output=64, activation_fn=tf.nn.relu)
            # fc3 = utils.fc(fc2, n_output=16, activation_fn=tf.nn.relu)
            fc1 = utils.lkrelu(utils.fc(self.state_input, 'fc1', nh=32, init_scale=np.sqrt(3.0)))
            fc2 = utils.lkrelu(utils.fc(fc1, 'fc2', nh=64, init_scale=np.sqrt(3.0)))
            fc3 = utils.lkrelu(utils.fc(fc2, 'fc3', nh=16, init_scale=np.sqrt(3.0)))
            # self.vf = utils.fc(fc3,1,activation_fn=tf.nn.relu)

            self.q_values = utils.lkrelu(utils.fc(fc3, 'q_values', nh=self.action_size, init_scale=np.sqrt(3.0)))
            # self.q_values = utils.fc(fc3, self.action_size, activation_fn=utils.lkrelu) # 每一个动作的q_value
            self.pi = tf.nn.softmax(self.q_values)
            self.values = tf.reduce_sum(tf.multiply(self.pi,self.q_values),axis=1)
            # self.pi = utils.fc(fc3,self.action_size,activation_fn=tf.nn.softmax) # 动作分布
            #self.pi = tf.nn.softmax(utils.fc(self.q_values, 'pi', nh=self.action_size, init_scale=np.sqrt(2.0)))
            # self.pi = tf.nn.softmax(utils.lkrelu(utils.fc(fc3, 'pi', nh=self.action_size, init_scale=np.sqrt(2.0))))
            # self.vf = tf.reduce_sum(tf.multiply(self.q_values,self.pi),1)
            # 动作用one-hot编码
            self.action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0) # 将输入的action编码成one-hot
            # 预测的q，根据当前的action预测动作
            self.q_value_pred = tf.reduce_sum(self.q_values * self.action_mask, 1)
            # self.q_value_pred = utils.fc(fc3,n_output=1,activation_fn=tf.nn.tanh)
            # q network的loss
            self.dqn_loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, self.q_value_pred)))
            self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.pi), 1)))
            # self.policy_loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=)
            self.loss = self.value_coef * self.dqn_loss + self.policy_coef * self.policy_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            self.train_op_dqn = self.optimizer.minimize(self.dqn_loss,global_step=self.global_step)

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

    def learn(self, buffer, batch_size):
        if buffer.size() <= batch_size:
            print("buffer size:",buffer.size())
            # loss = None
            return None,None,self.sess.run(self.global_step)
        else:
            loss = 0
            dloss = 0
            num_steps = int(buffer.size()//batch_size)
            # old_probs, old_v = self.policy_value_net.policy_value(state_batch)
            # for step in range(num_steps):
            for step in range(num_steps):
                # print("第{}次更新".format(step))
                # states0,next_states1,actions2,rewards3,dones4,mcts_probs5,values6
                # zip(states0,next_states1,actions2,rewards3,dones4,mcts_probs5,values6)
                minibatch = buffer.get_batch(batch_size=batch_size)
                state_batch = [data[0] for data in minibatch]
                next_state_batch = [data[1] for data in minibatch]
                action_batch = [data[2] for data in minibatch]
                # reward_batch = [data[3] for data in minibatch]
                done_batch = [data[4] for data in minibatch]
                mcts_prob_batch = [data[5] for data in minibatch]
                # value_batch = [data[6] for data in minibatch]
                # state_batch = [data[0] for data in minibatch]
                # action_batch = [data[1] for data in minibatch]
                # reward_batch = [data[2] for data in minibatch]
                # next_state_batch = [data[3] for data in minibatch]
                # done_batch = [data[4] for data in minibatch]
                state_batch = np.array(state_batch).reshape((-1,self.state_size))
                next_state_batch = np.array(next_state_batch).reshape((-1,self.state_size))
                action_batch = np.array(action_batch).reshape((batch_size))
                mcts_prob_batch = np.array(mcts_prob_batch).reshape((-1,self.action_size))
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
                old_probs, old_v = self.policy_value(state_batch[0])
                # print("mcts数据更新")
                # 最小化TD-error,即训练
                l, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.state_input: state_batch,
                    self.target_q: target_q,
                    self.action: action_batch,
                    self.mcts_probs: mcts_prob_batch,
                    self.learning_rate:self.lr*self.lr_multiplier
                })
                # states0,next_states1,actions2,rewards3,dones4,mcts_probs5,values6,p_states7,p_next_states8,p_actions9,p_rewards10,p_dones11

                # p_state_batch = [data[7] for data in minibatch]
                # p_next_state_batch = [data[8] for data in minibatch]
                # p_action_batch = [data[9] for data in minibatch]
                #
                # p_state_batch = np.array(p_state_batch).reshape((-1, self.state_size))
                # p_next_state_batch = np.array(p_next_state_batch).reshape((-1, self.state_size))
                # p_action_batch = np.array(p_action_batch).reshape((batch_size))
                # p_q_values = self.sess.run(self.q_values, feed_dict={self.state_input: p_next_state_batch})
                # p_max_q_values = p_q_values.max(axis=1)
                # p_target_q = np.array(
                #     [data[10] + self.gamma * p_max_q_values[i] * (1 - data[11]) for i, data in enumerate(minibatch)]
                # )
                # print(target_q)
                # print("dqn数据更新")
                target_q = target_q.reshape([batch_size])
                dl,_ = self.sess.run([self.dqn_loss,self.train_op_dqn], feed_dict={
                    self.state_input: state_batch,
                    self.target_q: target_q,
                    self.action: action_batch,
                    self.learning_rate:self.lr*self.lr_multiplier
                })

                new_probs, new_v = self.policy_value(state_batch[0])
                kl = np.mean(np.sum(old_probs * (old_probs - new_probs + 1e-10), axis=1))
                self.kl_targ = 0.02
                if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                    self.lr_multiplier /= 1.5
                elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                    self.lr_multiplier *= 1.5
                # l, _ ,am= self.sess.run([self.loss, self.train_op,self.action_mask], feed_dict={
                #     self.state_input: state_batch,
                #     self.target_q: target_q,
                #     self.action: action_batch
                # })
                # print("action",action_batch)
                # print("one hot",am)
                global_step = self.sess.run(self.global_step)
                self.mylogger.write_summary_scalar(global_step,"loss",l)
                # print(l)
                loss += l
                dloss += dl
            return loss/num_steps,dloss,global_step

    def policy_value(self,state):
        """
        输入:a batch of states
        :param state:
        :return: a batch of action probabilities and state values
        """
        # log_act_probs,value = self.sess.run([self.pi,self.q_value_pred],feed_dict={self.state_input: [state]})
        # act_probs = np.exp(log_act_probs)
        act_probs, q,values= self.sess.run([self.pi, self.q_values,self.values], feed_dict={self.state_input: [state]})
        # print("act_probs:{}, q:{}".format(act_probs,q))
        # act_probs = np.exp(log_act_probs)
        #act_probs = log_act_probs
        # value = np.sum(np.multiply(act_probs,q),axis=1)
        '''测试代码'''
        # log_act_probs,q_values,vf = self.sess.run([self.q_values,self.pi,self.vf],feed_dict={self.state_input: [state]})
        # print("测试:",log_act_probs,q_values,vf)
        # print(values)
        return act_probs, values

    def policy_value_fn(self, action, state):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        act_probs, value = self.policy_value(state[0])
        #print("act_probs, value",act_probs, value)
        #print("action:",act_probs.argmax())
        #print("action_:",max(zip(action, act_probs[0]),key=lambda act_prob:act_prob[1])[0])
        return zip(action, act_probs[0]), value[0]

    def get_optimal_action(self,action,state):
        """
        获得最优的action
        :param state:
        :return:
        """
        act_probs = self.sess.run(self.pi, feed_dict={self.state_input: [state]})
        # act_probs = np.exp(log_act_probs)
        # return act_probs.argmax()
        return max(zip(action, act_probs[0]),key=lambda act_prob:act_prob[1])[0]
