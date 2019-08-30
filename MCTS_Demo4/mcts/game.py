# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
from mcts.utils import get_true_action,get_recoverOb
# from utils import get_true_action,get_recoverOb
class Game(object):
    def __init__(self, env, **kwargs):
        self.env = env
        #self.moves = []


    def start_self_play(self, player, temp=1e-3):
        """
        用mcts获取数据
        :param player:
        :param temp:
        :return:
        """
        states,next_states,actions,rewards,dones,mcts_probs,values = [],[],[],[],[],[],[]
        while True:
            # 用mcts获取一个action以及其分布,v_value,rootOb
            action, action_probs, v_value, rootOb = player.get_action(Dirichlet_coef=0.3, temp=temp)
            # print(action)
            true_action = get_true_action(action)
            # print("action from mcts:",true_action)
            # 用mcts采样
            actions.append([true_action[0],true_action[1]])
            states.append(rootOb[7:])
            mcts_probs.append(action_probs)
            values.append(v_value)
            # 只恢复状态
            self.env.recover(get_recoverOb(rootOb))
            # 执行action
            # print(get_true_action(action))
            next_state , reward, done, _ = self.env.step(true_action)
            dones.append(done)
            next_states.append(next_state[7:])
            rewards.append(reward)
            if done or len(states)>3:
                print(done,len(actions))
                player.reset_player()
                return zip(states, mcts_probs, values,actions,next_states,rewards,dones)

