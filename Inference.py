import numpy as np
from collections import deque
from SimpleEnv import SnakeEnv
from SimpleGame import Game
from SimpleAlphaZero import MCTSPlayer
from SimplePolicyValueNet import PolicyValueNet


class Inference_Pipeline:

    def __init__(self, gameSpeed=1, train_model=True):
        # 初始化游戏环境
        self.Env = SnakeEnv(gameSpeed=gameSpeed, train_model=train_model)

        ob_space = self.Env.observation_space
        ac_space = self.Env.action_space
        nbatch = 1
        save_path = "snake713_1/"
        self.policy_value_net = PolicyValueNet(ob_space=ob_space,
                                               ac_space=ac_space,
                                               nbatch=nbatch,
                                               save_path=save_path,
                                               reuse=False)

    def run(self):
        ob = self.Env.reset()

        while (1):
            action_probs, _ = self.policy_value_net.policy_value_fn(self.Env.acts, ob)
            action_prob = max(action_probs, key=lambda act_prob: act_prob[1])
            # print(action_prob[0])
            ob, _, done, _ = self.Env.step(action_prob[0])

            """调试："""
            print(("ob  :{},  "
                   "done :{},  "
                   "action :{}"
                   ).format(ob, done, action_prob[0]))

if __name__=="__main__":
    inference = Inference_Pipeline(gameSpeed=0,train_model=False)
    inference.run()