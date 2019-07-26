# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import random
import numpy as np
from replay_buffer import ReplayBuffer
from game import Game
from alphaZero import MCTSPlayer
from policy_value_network import dqn
from env.simple_env import SnakeEnv


class TrainPipeline(object):
    def __init__(self,trainSpeed = 0,train_model = False):
        self.env = SnakeEnv(trainSpeed,train_model=train_model)
        self.game = Game(self.env)
        self.buffer_size = 16384
        self.batch_size = 512
        self.buffer = ReplayBuffer(self.buffer_size)
        self.save_path = "MctsModel/"
        self.epoch_num = 10000000
        # mcts需要的参数
        self.c_puct = 5
        self.n_playout = 512
        self.temp = 1.0
        # dqn需要的参数定义
        self.update_num_steps = 16 # 一个epoch的update_num_steps
        self.epsilon = 0.9
        self.epsilon_anneal = 0.01
        self.end_epsilon = 0.1
        self.lr = 0.001
        self.gamma = 0.9
        self.state_size = 3
        self.action_size = 4
        self.name_scope = 'dqn'
        self.policy_value_net = dqn(epsilon = self.epsilon,
                                    epsilon_anneal = self.epsilon_anneal,
                                    end_epsilon = self.end_epsilon,
                                    lr = self.lr,
                                    gamma = self.gamma,
                                    state_size = self.state_size,
                                    action_size = self.action_size,
                                    name_scope = self.name_scope,
                                    save_path=self.save_path)

        self.mcts_player = MCTSPlayer(self.env,
                                      self.policy_value_net.policy_value_fn,
                                      c_puct = self.c_puct,
                                      n_playout = self.n_playout,
                                      is_selfplay = 1)

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            # print(i)
            play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            # print(list(play_data))
            #print(type(play_data))
            play_data = list(play_data)[:]
            reward = np.sum([data[3] for data in play_data])
            # print(reward)
            #print(play_data)
            #print(type(play_data[0]))
            self.episode_len = len(play_data)
            self.buffer.add(play_data)
            # print("收集数据结束,buffer大小为{}".format(self.buffer.size()))
            return reward
            #print(self.buffer.size())

    def policy_updata(self):
        loss = self.policy_value_net.learn(self.buffer,num_steps=self.update_num_steps,batch_size=self.batch_size)
        return loss

    def run(self):
        try:
            for i in range(self.epoch_num):
                # print("第{}个epoch".format(i))
                r = self.collect_selfplay_data()
                l = self.policy_updata()
                if i % 50 == 0 and i >0:
                    self.policy_value_net.saveModel()
                print("loss:{},reward:{}".format(l,r))
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == "__main__":
    training_pipeline = TrainPipeline(trainSpeed=0, train_model=False)
    training_pipeline.run()