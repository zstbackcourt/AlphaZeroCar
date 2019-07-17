# -*- coding:utf-8 -*-
"""
ray train
@author: Weijie Shen
"""
from __future__ import print_function
import random
import numpy as np
from collections import deque
from SimpleEnv import SnakeEnv
from SimpleGame import Game
from SimpleAlphaZero import MCTSPlayer
from SimplePolicyValueNet import PolicyValueNet
import ray
from utils import create_Env,MPItrain
import time
import copy
from gym import spaces


@ray.remote
class Sampler(object):
    def __init__(self,rank):
        self.Env = create_Env(gameSpeed = 0,train_model=True)
        print("成功创建第{}个环境".format(rank))
        self.game = Game(self.Env)
        self.buffer_size = 4096
        self.n_playout = 512
        self.c_puct = 5
        ob_space = self.Env.observation_space
        ac_space = self.Env.action_space
        self.batch_size = 256
        save_path = "snake713_1/"
        nbatch = self.batch_size
        self.temp = 1.0
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.policy_value_net = PolicyValueNet(ob_space=ob_space,
                                               ac_space=ac_space,
                                               nbatch=nbatch,
                                               save_path=save_path,
                                               reuse=False,
                                               policy_coef=1.0,
                                               value_coef=10.0,
                                               l2_coef=1.0
                                               )

        self.mcts_player = MCTSPlayer(self.Env,
                                      self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
    def collectData(self):
        play_data = self.game.start_self_play(self.mcts_player,temp=self.temp)

        play_data = list(play_data)[:]
        # print(play_data)

        """有多少个（states, mcts_probs,value）"""
        self.episode_len = len(play_data)
        self.data_buffer.extend(play_data)
        buffer = self.data_buffer
        return buffer

    def get_network_weights(self):
        weights = self.policy_value_net.get_weights()
        return weights

    def set_network_weights(self,weights):
        self.policy_value_net.set_weights(weights)

@ray.remote
class ParameterServer(object):
    def __init__(self,ob_space,ac_space):
        self.mpitrain = MPItrain(learning_rate= 2e-3,
                 lr_multiplier= 1.0,
                 #temp= 1.0,
                 c_puct= 5,
                 n_playout= 512,
                 buffer_size= 4096,
                 batch_size=256,
                 play_batch_size= 1,
                 policy_coef= 1.0,
                 value_coef= 1.0,
                 l2_coef= 1.0,
                 epochs= 32,
                 kl_targ= 0.04,
                 check_freq= 1,
                 game_batch_num= 300000,
                 best_win_ratio= 0.0,
                 save_path="snake713_1/",
                 ob_space=ob_space,
                 ac_space=ac_space
                )
        # self.buffer = deque(maxlen=10000)

    # 测试用
    # def get_mpitrain_ob(self):
    #     return self.mpitrain.ob_space
    # def get_mpitrain_ac(self):
    #     return self.mpitrain.ac_space

    def rev_data(self,data):
        for i in range(len(data)):
            print("data from process{}".format(i))
            self.mpitrain.data_buffer.extend(ray.get(data[i]))
            self.updateNetwork()
            print("using data from process{} update network successfully".format(i))
        return True
            # print("buffer len:",len(self.mpitrain.data_buffer))

    def updateNetwork(self):
        # print("update the policy value network!")
        if len(self.mpitrain.data_buffer)>self.mpitrain.batch_size:
            self.mpitrain.update_policy_value_net()
            # loss, entropy, update_signal= self.mpitrain.update_policy_value_net()
            # print("loss:",loss,",entropy:",entropy)
            # return update_signal

    def get_weights(self):
        return self.mpitrain.get_policy_value_net_weights()

    def save_model(self):
        self.mpitrain.policy_value_net.save_model()
        print("saved the latest model successfully!")


if __name__ == "__main__":
    train_count = 0
    ray.init(num_cpus=24)
    env = create_Env()
    ac_space = env.action_space
    ob_space = env.observation_space
    # ob_space = copy.copy(env.observation_space)
    # ac_space = copy.copy(env.action_space)
    # # print(ob_space,ac_space)
    # if isinstance(ac_space, spaces.Box):
    #     act_dim = ac_space.shape[0]
    # elif isinstance(ac_space, spaces.Discrete):
    #     act_dim = ac_space.n
    # else:
    #     raise NotImplementedError
    #
    # if isinstance(ob_space, spaces.Box):
    #     ob_dim = ob_space.shape[0]
    # elif isinstance(ob_space, spaces.Discrete):
    #     ob_dim = ob_space.n
    # else:
    #     raise NotImplementedError
    ps = ParameterServer.remote(ob_space,ac_space)

    # env.close()
    # print("close",ob_space, ac_space)
    # ob = ps.get_mpitrain_ob.remote()
    # ac = ps.get_mpitrain_ac.remote()
    samplers = [Sampler.remote(sample_id) for sample_id in range(4)]
    signal = False
    # data = [sampler.collectData.remote() for sampler in samplers]
    # print(len(ray.get(data[0])))
    # ps.rev_data.remote(data)
    # signal_id = ps.updateNetwork.remote()
    # signal = ray.get(signal_id)
    # # print(signal)
    # if signal == True:
    #     psweights = ps.get_weights.remote()
    #     for sampler in samplers:
    #         weights = sampler.get_network_weights.remote()
    #         print("未赋值之前的网络参数:",ray.get(weights)["policyAndValue/pi_fc1/w"][0])
    #         sampler.set_network_weights.remote(psweights)
    #         weights = sampler.get_network_weights.remote()
    #         print("赋值之后的网络参数:", ray.get(weights)["policyAndValue/pi_fc1/w"][0])
    # # print(len(data))
    # signal = False
    while True:
        print("train_count:",train_count)
        train_count += 1
        data = [sampler.collectData.remote() for sampler in samplers]
        signal_id = ps.rev_data.remote(data)
        # signal_id = ps.updateNetwork.remote()
        signal = ray.get(signal_id)
        # print(signal)
        if signal == True:
            psweights = ps.get_weights.remote()
            for sampler in samplers:
                # weights = sampler.get_network_weights.remote()
                # print("未赋值之前的网络参数:",ray.get(weights)["policyAndValue/pi_fc1/w"][0])
                sampler.set_network_weights.remote(psweights)
                # weights = sampler.get_network_weights.remote()
                # print("赋值之后的网络参数:", ray.get(weights)["policyAndValue/pi_fc1/w"][0])
        # print(len(data))
        if train_count % 2 == 0:
            ps.save_model.remote()
        signal = False
