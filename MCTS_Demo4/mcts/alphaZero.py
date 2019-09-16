# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
from mcts.utils import softmax,get_true_action,get_recoverOb
# from utils import softmax,get_true_action,get_recoverOb
import numpy as np
import copy
import sys
sys.setrecursionlimit(100000)

class TreeNode(object):
    def __init__(self, parent, prior_p):
        """
        定义树节点
        :param parent:
        :param prior_p:
        """
        self._parent = parent # 父节点
        self._children = {} # 孩子
        self._n_visits = 0 # 节点被访问的次数
        self._Q = 0 # Q value
        self._u = 0 # u(s,a) puct
        self._P = prior_p # 该节点对应的动作的概率
        self._ob = None # 该节点保存的observation，就是状态

    def expand(self, action_priors):
        """
        给叶子节点expand孩子
        :param action_priors:
        :return:
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)  # 注意这里的初始化

    def select(self, c_puct):
        """
        根据puct选择一个孩子节点(选择一个action)
        :param c_puct:
        :return: 孩子节点
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        更新叶子节点的访问次数和Q value
        :param leaf_value:
        :return:
        """
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits


    def update_recursive(self, leaf_value):
        """
        调用update迭代更新所有的节点的信息
        :param leaf_value:
        :return:
        """
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        获得节点的value，Q+u
        :param c_puct:
        :return:
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """
        判断节点是不是叶子
        :return:
        """
        return self._children == {}

    def is_root(self):
        """
        判断节点是不是根节点
        :return:
        """
        return self._parent is None


class MCTS(object):

    def __init__(self, env, policy_value_fn, c_puct=5, n_playout=10000):
        """
        MCTS class
        :param env: 环境
        :param policy_value_fn: MCTS使用的用来评估节点的策略价值函数
        :param c_puct: puct系数
        :param n_playout: playout次数
        """
        self.env = env
        # 初始化根节点
        self._root = TreeNode(None, 0.0)
        self._root._ob = self.env.reset() # 这里是直接获得的ob，送入网络的时候要去掉前三维
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        # 用policy_value网络对root进行评估
        action_probs, leaf_value = self._policy_value_fn(self.env.acts, self._root._ob[4:].reshape(-1, self.env.ob_dim-4))
        if not isinstance(leaf_value, np.float32):
            print(leaf_value.type)
            raise ValueError("leaf_value的类型不对")

        a_p = list(copy.deepcopy(action_probs))
        # 给root expand孩子
        self._root.expand(action_probs)

    def _playout(self):
        """
        定义playout，一次playout就是从当前的root节点开始根据上限置信走到叶子节点，然后利用policy_value将节点展开
        然后迭代更新整个树
        :return:
        """
        c_node = self._root
        action = None

        while (1):
            """只要没有个走到叶子节点，就select节点继续往下走"""
            if c_node.is_leaf():
                break
            else:
                action, c_node = c_node.select(self._c_puct)

        # 走到了叶子节点，在环境中恢复该叶子节点的父节点保存的状态
        self.env.recover(get_recoverOb(c_node._parent._ob))
        # 在环境中执行该叶子节点对应的action，获得该叶子节点应该保存的状态
        # print(get_true_action(action))
        ob, reward, done, _ = self.env.step(get_true_action(action)) # 这里的ob也要去掉前两维
        if done == False:
            # 如果该叶子节点不是done，就保存状态
            c_node._ob = ob
            # 用policy_value对该叶子节点进行评估
            action_probs, leaf_value = self._policy_value_fn(self.env.acts, ob[4:].reshape(-1, self.env.ob_dim-4))
            if not isinstance(leaf_value,np.float32):
                # print(leaf_value.type)
                raise ValueError("leaf_value的类型不对")
            # 利用policy_value得到的action_probs（动作概率分布）来expand孩子
            c_node.expand(action_probs)
            # 更新整个树的节点信息
            c_node.update_recursive(leaf_value + reward)
        else:
            # print("done!",c_node._parent._ob[0],ob[0])
            c_node.update_recursive(reward)

    def get_move_probs(self, temp=1e-3):
        """
        获取当前root节点的最优action，action_probs，Q和保存的状态ob
        :param temp:
        :return:
        """
        for n in range(self._n_playout):
            self._playout()

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        #print("act_visits",act_visits)
        acts, visits = zip(*act_visits)
        #print("visits",visits)
        # q_u_value = [(node._Q, node._u) for act, node in self._root._children.items()]
        # q,u = zip(*q_u_value)
        act_probs = softmax(np.log(np.array(visits) + 1e-10) / temp)
        '''当前root节点对应的ob经过policy获得的action'''
        # policy_action_probs,policy_leaf_value= self._policy_value_fn(self.env.acts, self._root._ob.reshape(-1, self.env.ob_dim))
        # print("acts{},act_probs{},policy_action_probs{}:".format(acts,act_probs,policy_action_probs))
        return acts, act_probs, self._root._Q, self._root._ob

    def update_with_move(self, last_move):
        """
        更新当前的root（即推进一步）
        :param last_move:
        :return:
        """
        if last_move in self._root._children:
            # 从root节点的孩子中选择对应last_move这个动作的孩子作为新的root
            self._root = self._root._children[last_move]

            if (self._root._ob is None and self._root.is_leaf()):
                # 如果新的root的ob是None且该root是一个没有展开过的叶子节点
                # 就要用该root的父节点的状态来恢复环境，然后执行一步将新的状态
                # 保存在这个新的root节点上
                self.env.recover(get_recoverOb(self._root._parent._ob))
                ob, reward, done, _ = self.env.step(get_true_action(last_move))
                self._root._ob = ob
                # print("往前走了一步(expand)")
                # print(self._root._parent._ob[0])
                # print(last_move,self._root._ob[0])

                # 展开
                action_probs, leaf_value = self._policy_value_fn(self.env.acts,self._root._ob[4:].reshape(-1, self.env.ob_dim-4))
                self._root.expand(action_probs)
            self._root._parent = None
        else:
            # 一条轨迹结束，树重新展开
            self._root = TreeNode(None, 0.0)
            self._root._ob = self.env.reset()
            action_probs, leaf_value = self._policy_value_fn(self.env.acts, self._root._ob[4:].reshape(-1, self.env.ob_dim-4))
            self._root.expand(action_probs)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self,
                 env,
                 policy_value_function,
                 c_puct=5,
                 n_playout=20,
                 is_selfplay=0):
        """
        MCTS player class
        :param env:
        :param policy_value_function:
        :param c_puct:
        :param n_playout:
        :param is_selfplay:
        """
        """！！！！！！！"""
        self.policy_value_fn = policy_value_function
        # print("define MCTS player")
        self.mcts = MCTS(env, policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        """
        重置player，就是重置整个树
        :return:
        """
        self.mcts.update_with_move(-1)

    def get_action(self, temp=1e-3, Dirichlet_coef=0.1):
        """
        根据蒙特卡洛树获取动作
        :param temp:
        :param Dirichlet_coef:
        :return:动作action,概率分布,当前root节点的v_value,当前root节点的ob
        """
        acts, probs, v_value, rootOb = self.mcts.get_move_probs(temp=temp)


        if self._is_selfplay:
            # 用狄利克雷噪声选出一个动作
            action = np.random.choice(
                acts,
                p=(1.0 - Dirichlet_coef) * probs + Dirichlet_coef * np.random.dirichlet(0.3 * np.ones(len(probs)))
            )

            self.mcts.update_with_move(action)
        else:
            action = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        return action, probs, v_value, rootOb