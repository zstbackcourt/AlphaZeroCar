# -*- coding: utf-8 -*-
"""
"""

from __future__ import print_function


class Game(object):
    """game server"""
    def __init__(self, env, **kwargs):
        self.env = env
        self.moves = []

    def start_self_play(self, player, temp=1e-3):

        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs) for training
        """

        # ob = self.env.reset()

        states, mcts_probs, values = [], [], []

        while True:

            """move_probs"""
            """move选出的动作"""
            # print("moves length：",len(self.moves))
            # print("moves：",self.moves)

            action, action_probs, v_value, rootOb = player.get_action(Dirichlet_coef = 0.3,
                                                            temp=temp)
            # print(action)
            # print(action_probs)

            states.append(rootOb)  # 这个状态包括每个选手落子的轨迹、最新的落子、当前谁落子
            mcts_probs.append(action_probs)
            values.append(v_value)

            self.moves.append(action)

            """需要先返回状态再做这个动作"""
            self.env.recover(rootOb)  # 只恢复状态
            _ , _, done, _ = self.env.step(action)  # 只作动作，不恢复状态

            if done or len(self.moves)>300:
                # reset MCTS root node
                # print("done!!!")

                # self.env.reset()
                # for move in self.moves:
                #     self.env.step(move)

                player.reset_player()
                self.moves = []

                return zip(states, mcts_probs, values)