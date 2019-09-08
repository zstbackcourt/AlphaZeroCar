# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
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
from mcts.utils import get_true_action,get_recoverOb
import time
from baselines.common import explained_variance, set_global_seeds
sys.setrecursionlimit(100000)

class Inference_Pipeline:

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

        # self.policy_value_net = PolicyValueNet(ob_space=ob_space,
        #                                        ac_space=ac_space,
        #                                        nbatch=nbatch,
        #                                        save_path=save_path,
        #                                        policy_coef=1.0,
        #                                        value_coef=10.0,
        #                                        l2_coef=1.0,
        #                                        reuse=False)
        self.actions = [50,50,50]
        self.obs = []
    def run(self):
        ob = self.Env.reset()


        for i in range(len(self.actions)):
            ob, _, done, _ = self.Env.step(get_true_action(self.actions[i]))
            self.obs.append(ob)
        for i in range(len(self.obs)):
            print("{}:".format(i),self.obs[i][0:10])
        self.Env.recover(get_recoverOb(self.obs[1]))
        # print("recover ",ob[0:10])
        ob, _, done, _ = self.Env.step(get_true_action(self.actions[2]))
        print("recover ", ob[0:10])
        #     print(ob[0:10])
        # ob = self.Env.recover(get_recoverOb(self.obs[1]))
        # print(ob[0:10])
        # ob, _, done, _ = self.Env.step(get_true_action(self.actions[2]))
        # print(ob[0:10])
        # while (1):
            # action_probs, _ = self.policy_value_net.policy_value_fn(self.Env.acts, ob[6:].reshape(-1,self.ob_dim-6))
            # action_prob = max(action_probs, key=lambda act_prob: act_prob[1])
            # # print(action_prob[0])

            #ob, _, done, _ = self.Env.step(get_true_action())

            # """调试："""
            # print(("ob  :{},  "
            #        "done :{},  "
            #        "action :{}"
            #        ).format(ob, done, action_prob[0]))

if __name__ == '__main__':
    inference_pipline = Inference_Pipeline()
    inference_pipline.run()
    from tensorflow.contrib.image import transform


