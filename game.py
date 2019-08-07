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
        """
        用mcts获取数据
        :param player:
        :param temp:
        :return:
        """
        # mcts_probs是mcts获得的动作的分布，values是mcts做出的价值评估
        # states,next_states,actions,rewards,dones,mcts_probs,values,p_states,p_next_states,p_actions,p_rewards,p_dones= [],[],[],[],[],[],[],[],[],[],[],[]
        # sum_reward = 0
        states,next_states,actions,rewards,dones,mcts_probs,values = [],[],[],[],[],[],[]



        # states, mcts_probs, values,next_states,reward = [], [], [],[],[]

        while True:
            terminal_flag = 0
            # 用mcts获取一个action以及其分布,v_value,rootOb
            action, action_probs, v_value, rootOb,policy_acts_probs= player.get_action(Dirichlet_coef = 0.3,temp=temp)
            # print(action, action_probs, v_value, rootOb)

            p_action = max(policy_acts_probs, key=lambda act_prob: act_prob[1])[0]

            # 用mcts采样
            actions.append(action)
            states.append(rootOb)
            mcts_probs.append(action_probs)
            values.append(v_value)
            # ob, r, np.array(d), infos = env.step
            self.moves.append(action)
            self.env.recover(rootOb) # 只恢复状态
            #print("rootOb:{},recover后:{},preDis:{}".format(rootOb,self.env.snakePosition,self.env.preDis))
            next_state , reward, done, _ = self.env.step(action)
            if done:
                terminal_flag = 1

            #print("执行完:{},preDis:{}".format(next_state,self.env.preDis))
            #print(len(self.moves))
            dones.append(done)
            next_states.append(next_state)
            rewards.append(reward)

            # 直接用dqn采样
            actions.append(p_action)
            states.append(rootOb)
            mcts_probs.append(action_probs)
            self.moves.append(p_action)
            self.env.recover(rootOb)
            next_state, reward, done, _ = self.env.step(p_action)
            if done:
                terminal_flag = 1
            dones.append(done)
            next_states.append(next_state)
            rewards.append(reward)
            values.append(v_value)
            # p_actions.append(p_action)
            # p_states.append(rootOb)
            # self.moves.append(p_action)
            # self.env.recover(rootOb)  # 只恢复状态
            # p_next_state, p_reward, p_done, _ = self.env.step(p_action)
            # p_next_states.append(p_next_state)
            # p_rewards.append(p_reward)
            # p_dones.append(p_done)
            #print(action,next_state,reward,done)
            #print("P:",p_action,p_next_state, p_reward, p_done)
            #print("当前获得reawrd:",reward)
            # sum_reward += reward
            # print(len(states),len(next_states),len(actions),len(rewards),len(dones),len(mcts_probs),len(values))
            if terminal_flag == 1 or len(self.moves)>1024:
                # print("done!!!")
                player.reset_player()
                self.moves = []
                # states,next_states,actions,rewards,dones,mcts_probs,values
                # print("actions:",actions)
                return zip(states,next_states,actions,rewards,dones,mcts_probs,values)

    def play_with_policynetwork(self):
        # pass
        states, next_states, actions, rewards, dones, mcts_probs, values = [], [], [], [], [], [], []
        pass
