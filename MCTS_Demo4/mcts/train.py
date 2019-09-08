# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Shen Weijie
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from mcts.env import UnityEnv
from mcts.game import Game
from mcts.alphaZero import MCTSPlayer
from mcts.policy_value_network import PolicyValueNet  # Tensorflow
from mcts.ddpg import ddpg
# from env import UnityEnv
# from game import Game
# from alphaZero import MCTSPlayer
# from policy_value_network import PolicyValueNet  # Tensorflow
# from ddpg import ddpg
import sys
from mcts.utils import get_true_action
# from utils import get_true_action
import time
from baselines.common import explained_variance, set_global_seeds
sys.setrecursionlimit(100000)

class TrainPipeline():

    def __init__(self):

        # 初始化游戏环境
        self.Env = UnityEnv(env_directory=None, worker_id=0, train_model=True, no_graphics=True)

        # 初始化游戏
        self.game = Game(self.Env)

        # training params
        self.learning_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL 基于KL自适应调整学习速度
        self.temp = 1.0  # the temperature param 论文中的

        self.c_puct = 5

        # buffer params
        self.n_playout = 256  # num of simulations for each move 在每一步总的模拟数
        self.buffer_size = 65536
        self.batch_size = 256  # mini-batch size for training

        self.data_buffer = deque(maxlen=self.buffer_size)  # 双端队列，队列满了之后，会将最开始加入的删掉

        self.play_batch_size = 1

        # tensorflow模型训练参数
        self.policy_coef = 1.0
        self.value_coef = 1.0
        self.l2_coef = 1.0

        self.epochs = 32  # num of train_steps for each update
        self.kl_targ = 0.04
        self.check_freq = 1

        self.game_batch_num = 300000
        self.best_win_ratio = 0.0

        ob_space = self.Env.observation_space
        ac_space = self.Env.action_space
        self.ob_dim = self.Env.ob_dim
        self.act_dim = self.Env.act_dim
        nbatch = self.batch_size
        nsteps = 1

        save_path = "MCTS_Car/"

        # set_global_seeds(0)

        self.policy_value_net = PolicyValueNet(ob_space=ob_space,
                                               ac_space=ac_space,
                                               nbatch=nbatch,
                                               save_path=save_path,
                                               policy_coef=1.0,
                                               value_coef=10.0,
                                               l2_coef=1.0,
                                               reuse=False)

        self.mcts_player = MCTSPlayer(self.Env,
                                      self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        self.ddpg_net = ddpg(ob_dim=self.ob_dim , nbatch=nbatch, save_path="./DDPG_Car/model",
                             batch_size=self.batch_size,
                             gamma=0.99, a_lr=0.0001, c_lr=0.0001, tau=0.001, )

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)

            play_data = list(play_data)[:]
            # print(play_data)

            """有多少个（states, mcts_probs,value）"""
            self.episode_len = len(play_data)

            self.data_buffer.extend(play_data)  # 双端队列

    def policy_update(self):
        """update the policy-value net"""
        """从data_buffer中随机选出batch_size长度的数据"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        value_batch = [data[2] for data in mini_batch]
        # print(state_batch)
        state_batch = np.array(state_batch).reshape(self.batch_size, self.ob_dim-7)
        mcts_probs_batch = np.array(mcts_probs_batch).reshape(self.batch_size, self.act_dim)
        value_batch = np.array(value_batch).reshape(self.batch_size, 1)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            _loss, _value_loss, _policy_loss, _l2_penalty, _entropy = self.policy_value_net.train_step(state_batch,
                                                                                                       mcts_probs_batch,
                                                                                                       value_batch,
                                                                                                       self.learning_rate * self.lr_multiplier)
            ddpg_loss = self.ddpg_net.learn(mini_batch)
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
               "value_loss:{},  "
               "policy_loss:{},  "
               "l2_penalty:{},  "
               "entropy:{},"
               "ddpg_loss:{}  "
               ).format(kl,
                        self.lr_multiplier,
                        _loss,
                        _value_loss,
                        _policy_loss,
                        _l2_penalty,
                        _entropy,
                        ddpg_loss))

        return _loss, _value_loss, _policy_loss, _l2_penalty, _entropy

    def run(self):
        try:
            number = 0
            for i in range(self.game_batch_num):  # 1500局游戏

                self.collect_selfplay_data(self.play_batch_size)  # n_game = 1

                # print("batch i:{}, episode_len:{}".format(i+1, self.episode_len)) #这局游戏多长

                # print("data_buffer length: ", len(self.data_buffer))

                if len(self.data_buffer) >= self.batch_size:
                    # print("开始更新网络！！！")
                    number += 1
                    print("第", number, "轮：")
                    self.policy_update()

                    if (i + 1) % self.check_freq == 0:
                        self.policy_value_net.save_model()
                        self.ddpg_net.save_model()
                if len(self.data_buffer)>=65535:
                    self.data_buffer.clear()
                    # print("清空队列")
        except KeyboardInterrupt:
            print('\n\rquit')


class Inference_Pipeline():

    def __init__(self):
        # 初始化游戏环境
        self.Env = UnityEnv(env_directory=None, worker_id=0, train_model=False, no_graphics=True)

        ob_space = self.Env.observation_space
        ac_space = self.Env.action_space
        self.ob_dim = self.Env.ob_dim
        self.act_dim = self.Env.act_dim
        nbatch = 1
        save_path = "MCTS_Car/"

        # set_global_seeds(0)

        self.policy_value_net = PolicyValueNet(ob_space=ob_space,
                                               ac_space=ac_space,
                                               nbatch=nbatch,
                                               save_path=save_path,
                                               policy_coef=1.0,
                                               value_coef=10.0,
                                               l2_coef=1.0,
                                               reuse=False)

    def run(self):
        ob = self.Env.reset()

        while (1):
            action_probs, _ = self.policy_value_net.policy_value_fn(self.Env.acts, ob[7:].reshape(-1,self.ob_dim-7))
            action_prob = max(action_probs, key=lambda act_prob: act_prob[1])
            print(ob[7:],action_prob[0],get_true_action(action_prob[0]))
            # print(get_true_action(action_prob[0]))
            ob, _, done, _ = self.Env.step(get_true_action(action_prob[0]))

            # """调试："""
            # print(("ob  :{},  "
            #        "done :{},  "
            #        "action :{}"
            #        ).format(ob, done, action_prob[0]))

class DDPG_Inference():
    def __init__(self):
        self.Env = UnityEnv(env_directory=None, worker_id=0, train_model=False, no_graphics=True)

        ob_space = self.Env.observation_space
        ac_space = self.Env.action_space
        self.ob_dim = self.Env.ob_dim
        self.act_dim = self.Env.act_dim
        nbatch = 1
        save_path = "DDPG_Car/"
        self.ddpg_net = ddpg(ob_dim=self.ob_dim, nbatch=nbatch, save_path="./DDPG_Car",
                             batch_size=1,
                             gamma=0.99, a_lr=0.0001, c_lr=0.0001, tau=0.001, )

    def run(self):
        ob = self.Env.reset()
        # ob[7:].reshape(self.ob_dim-7)
        while (1):
            action= self.ddpg_net.actor.get_action(ob[7:].reshape([-1,self.ob_dim-7]),self.ddpg_net.sess)
            print(action)
            # action_prob = max(action_probs, key=lambda act_prob: act_prob[1])
            # print(action_prob[0])
            # ob, _, done, _ = self.Env.step(get_true_action(action_probs))


if __name__ == '__main__':

    train_model = False  # use mcts
    #
    # train_model = False  # network inference

    if train_model:
        training_pipeline = TrainPipeline()
        training_pipeline.run()
    else:
        inference_pipline = Inference_Pipeline()
        inference_pipline.run()
        # inference_pipline = DDPG_Inference()
        # inference_pipline.run()