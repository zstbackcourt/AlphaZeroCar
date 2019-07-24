# -*- coding:utf-8 -*-
"""
利用蒙特卡洛树收集数据
@author: Weijie Shen
"""


class Game(object):
    def __init__(self, env, **kwargs):
        self.env = env
        self.moves = []

    def start_self_play(self, player, temp=1e-3):
        # mcts_probs是mcts获得的动作的分布，values是mcts做出的价值评估
        states,next_states,actions,rewards,dones,mcts_probs,values = [],[],[],[],[],[],[]




        # states, mcts_probs, values,next_states,reward = [], [], [],[],[]

        while True:
            # 用mcts获取一个action以及其分布,v_value,rootOb
            action, action_probs, v_value, rootOb = player.get_action(Dirichlet_coef = 0.3,temp=temp)
            # print(action, action_probs, v_value, rootOb)
            actions.append(action)
            states.append(rootOb)
            mcts_probs.append(action_probs)
            values.append(v_value)
            # ob, r, np.array(d), infos = env.step
            self.moves.append(action)
            self.env.recover(rootOb) # 只恢复状态
            next_state , reward, done, _ = self.env.step(action)
            dones.append(done)
            next_states.append(next_state)
            rewards.append(reward)
            #print(len(states),len(next_states),len(actions),len(rewards),len(dones),len(mcts_probs),len(values))
            if done or len(self.moves)>5:
                print("done!!!")
                player.reset_player()
                self.moves = []
                # states,next_states,actions,rewards,dones,mcts_probs,values
                return zip(states,next_states,actions,rewards,dones,mcts_probs,values)