# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
from policy_value_network import dqn
from env.simple_env import SnakeEnv
# from game import Game
from alphaZero import MCTSPlayer


class InferencePipeline(object):
    def __init__(self,trainSpeed = 5,train_model = False):
        self.env = SnakeEnv(trainSpeed, train_model=train_model)
        # self.game = Game(self.env)
        self.c_puct = 5
        self.n_playout = 512
        self.temp = 1.0
        self.state_size = 3
        self.action_size = 4
        self.lr = 0.001
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_anneal = 0.01
        self.end_epsilon = 0.1
        self.name_scope = 'dqn'
        self.save_path = "MctsModel/"
        self.policy_value_net = dqn(epsilon=self.epsilon,
                                    epsilon_anneal=self.epsilon_anneal,
                                    end_epsilon=self.end_epsilon,
                                    lr=self.lr,
                                    gamma=self.gamma,
                                    state_size=self.state_size,
                                    action_size=self.action_size,
                                    name_scope=self.name_scope,
                                    save_path=self.save_path)

        self.mcts_player = MCTSPlayer(self.env,
                                      self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def run(self):
        ob = self.env.reset()
        while True:
            # action, _, _, _ = self.mcts_player.get_action(Dirichlet_coef=0.0, temp=self.temp)
            # print(action)
            # action = self.policy_value_net.get_optimal_action(ob[0])
            action_probs , _ = self.policy_value_net.policy_value_fn(self.env.acts,ob)
            # print(action_probs)
            action_prob = max(action_probs,key=lambda act_prob:act_prob[1])
            print(action_prob[0],action_prob)
            ob,_,done,_ = self.env.step(action_prob[0])
            #       'done:{},'
            #       'action:{}').format(ob,done,action[0]))

if __name__ == '__main__':
    inference_pipeline = InferencePipeline()
    inference_pipeline.run()