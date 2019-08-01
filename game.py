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
        # sum_reward = 0



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
            #print("rootOb:{},recover后:{},preDis:{}".format(rootOb,self.env.snakePosition,self.env.preDis))
            next_state , reward, done, _ = self.env.step(action)
            #print("执行完:{},preDis:{}".format(next_state,self.env.preDis))
            #print(len(self.moves))
            dones.append(done)
            next_states.append(next_state)
            rewards.append(reward)
            #print("当前获得reawrd:",reward)
            # sum_reward += reward
            # print(len(states),len(next_states),len(actions),len(rewards),len(dones),len(mcts_probs),len(values))
            if done or len(self.moves)>512:
                # print("done!!!")
                player.reset_player()
                self.moves = []
                # states,next_states,actions,rewards,dones,mcts_probs,values
                # print("actions:",actions)
                return zip(states,next_states,actions,rewards,dones,mcts_probs,values)