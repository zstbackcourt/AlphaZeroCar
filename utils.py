# -*- coding:utf-8 -*-
"""
utils脚本
@author: Weijie Shen
"""
from SimpleEnv import SnakeEnv
from collections import deque
import random
import numpy as np
from SimplePolicyValueNet import PolicyValueNet




class MPItrain():
    def __init__(self,
                 learning_rate,
                 lr_multiplier,
                 # temp,
                 c_puct,
                 n_playout,
                 buffer_size,
                 batch_size,
                 play_batch_size,
                 policy_coef,
                 value_coef,
                 l2_coef,
                 epochs,
                 kl_targ,
                 check_freq,
                 game_batch_num,
                 best_win_ratio,
                 save_path,
                 ob_space,
                 ac_space):

        self.learning_rate = learning_rate
        self.lr_multiplier = lr_multiplier  # adaptively adjust the learning rate based on KL 基于KL自适应调整学习速度
       #  self.temp = temp  # the temperature param 论文中的
        self.c_puct = c_puct

        # buffer params
        self.n_playout = n_playout  # num of simulations for each move 在每一步总的模拟数
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # mini-batch size for training

        self.data_buffer = deque(maxlen=self.buffer_size)  # 双端队列，队列满了之后，会将最开始加入的删掉

        self.play_batch_size = play_batch_size

        # tensorflow模型训练参数
        self.policy_coef = policy_coef
        self.value_coef = value_coef
        self.l2_coef = l2_coef

        self.epochs = epochs  # num of train_steps for each update
        self.kl_targ = kl_targ
        self.check_freq = check_freq

        self.game_batch_num = game_batch_num
        self.best_win_ratio = best_win_ratio
        self.save_path = save_path
        #print(ob_space,ac_space)
        self.ob_space = ob_space
        self.ac_space = ac_space
        #print(self.ob_space,self.ac_space)
        self.policy_value_net = PolicyValueNet(ob_space=self.ob_space,
                                               ac_space=self.ac_space,
                                               nbatch=self.batch_size,
                                               save_path=self.save_path,
                                               reuse=False)

        # # training params
        # self.learning_rate = 2e-3
        # self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL 基于KL自适应调整学习速度
        # self.temp = 1.0  # the temperature param 论文中的
        # self.c_puct = 5
        #
        # # buffer params
        # self.n_playout = 512  # num of simulations for each move 在每一步总的模拟数
        # self.buffer_size = 10000
        # self.batch_size = 128  # mini-batch size for training
        #
        # self.data_buffer = deque(maxlen=self.buffer_size)  # 双端队列，队列满了之后，会将最开始加入的删掉
        #
        # self.play_batch_size = 1
        #
        # # tensorflow模型训练参数
        # self.policy_coef = 1.0
        # self.value_coef = 1.0
        # self.l2_coef = 1.0
        #
        # self.epochs = 32  # num of train_steps for each update
        # self.kl_targ = 0.02
        # self.check_freq = 1
        #
        # self.game_batch_num = 300000
        # self.best_win_ratio = 0.0
        #
        # ob_space = self.Env.observation_space
        # ac_space = self.Env.action_space
        #
        # nbatch = self.batch_size
        # nsteps = 1
        #
        # save_path = "MCTS_snake/model"
        #
        # # set_global_seeds(0)
        #
        # self.policy_value_net = PolicyValueNet(ob_space=ob_space,
        #                                        ac_space=ac_space,
        #                                        nbatch=nbatch,
        #                                        save_path=save_path,
        #                                        reuse=False)

    # 获取当前网络的参数
    def get_policy_value_net_weights(self):
        return self.policy_value_net.get_weights()

    def update_policy_value_net(self):

        """update the policy-value net"""
        """从data_buffer中随机选出batch_size长度的数据"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        value_batch = [data[2] for data in mini_batch]

        state_batch = np.array(state_batch).reshape(self.batch_size, 3)
        mcts_probs_batch = np.array(mcts_probs_batch).reshape(self.batch_size, 4)
        value_batch = np.array(value_batch).reshape(self.batch_size, 1)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            _loss, _value_loss, _policy_loss, _l2_penalty, _entropy = self.policy_value_net.train_step(state_batch,
                                                                                                       mcts_probs_batch,
                                                                                                       value_batch,
                                                                                                       self.learning_rate * self.lr_multiplier)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            """kl散度，使用熵和交叉熵来计算得，描述两个概率分布时间的差异"""
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            self.kl_targ = 0.02
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                # print("KL过大，停止更新")
                break

        # adaptively adjust the learning rate
        """
        如果kl比较大，调小学习率；
        如果kl比较小，调大学习率；

        """
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained_var_old = (1 -
        #                      np.var(np.array(value_batch) - old_v.flatten()) /
        #                      (np.var(np.array(value_batch) + 1e-8)))
        # explained_var_new = (1 -
        #                      np.var(np.array(value_batch) - new_v.flatten()) /
        #                      (np.var(np.array(value_batch)+1e-8)))

        print(("kl:{},  "
               "lr_multiplier:{},  "
               "loss:{},  "
               "value_loss:{}  "
               "policy_loss:{}  "
               "l2_penalty:{}  "
               "entropy:{},  "
               ).format(kl,
                        self.lr_multiplier,
                        _loss,
                        _value_loss,
                        _policy_loss,
                        _l2_penalty,
                        _entropy))

        return _loss, _value_loss, _policy_loss, _l2_penalty, _entropy









def create_Env(gameSpeed=0,train_model=True):
    """
    Create a Unity Env
    :param dir: Unity file name e.g. MCTS_003
    :param rank: process ID using as Unity Env's worker_id
    :param model: whether Train or Inference
    :param nographic: whether using graphic
    :return: a object of Unity env
    """
    env = SnakeEnv(gameSpeed=gameSpeed,train_model=train_model)
    return env