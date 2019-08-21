# -*- coding:utf-8 -*-
"""
Unity的环境
@author: Weijie Shen
"""
import numpy as np
import gym
import os
from gym import error, spaces
from mlagents.envs import UnityEnvironment



class UnityEnv(gym.Env):
    def __init__(self, env_directory, worker_id, train_model=True, no_graphics=False):
        self.env = UnityEnvironment(file_name=env_directory, worker_id=worker_id, seed=1,no_graphics=no_graphics)
        self.train_model = train_model
        self.num_envs = 1
        self.envs = 1
        self.brain_name = self.env.brain_names[0]
        self.acts = [i for i in range(121)] # 离散化后的动作-1~1一共11*11=121个组合
        self.action_space = spaces.Box(low=np.zeros([121]) - 1, high=np.zeros([121]) + 1, dtype=np.float32)
        # 状态空间一共是243维，其中0，1两维不送入policy是用来在Unity中恢复状态的
        self.observation_space = spaces.Box(low=np.zeros([243]) - 1, high=np.zeros([243]) + 1, dtype=np.float32)
        self.act_dim = 121
        self.ob_dim = 243

    def seed(self, seed=None):
        """
        :param seed:
        :return:
        """
        pass

    def step(self, a):
        action = {}
        infos = {}
        action[self.brain_name] = a
        info = self.env.step(vector_action=action)
        brainInfo = info[self.brain_name]
        reward = np.array(brainInfo.rewards)
        done = np.array(brainInfo.local_done)
        ob = np.array(brainInfo.vector_observations)
        return ob[-1, 0:243], reward, done[0], infos

    def reset(self):
        info = self.env.reset(train_mode=self.train_model)
        brainInfo = info[self.brain_name]
        ob = np.array(brainInfo.vector_observations)
        return ob[-1, 0:243]

    def recover(self,a):
        self.step(a)


    def render(self):
        '''
        :return: None
        '''
        pass

    def close(self):
        '''
        :return: None
        '''
        self.env.close()