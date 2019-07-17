# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    """树的节点：保存Q值、先验概率P、访问次数"""
    """搜索树中的每个边(s, a)存储先验概率P(s, a)，访问计数N(s, a)和行动价值Q(s, a)"""
    """使置信上限区间𝑄(𝑠,𝑎)+𝑈(𝑠,𝑎)最大化的走子，其中𝑈(𝑠,𝑎)∝𝑃(𝑠,𝑎)/(1+𝑁(𝑠,𝑎))，"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

        self._ob = None

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # print(action_priors)
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)  # 注意这里的初始化
        #         print(action,prob)
        #
        # print("-------")

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        # print(max(self._children.items(),
        #            key=lambda act_node: act_node[1].get_value(c_puct)))
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

        # print(self._n_visits)

    def update_recursive(self, leaf_value):  # 递归地更新
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        # print(self._P)

        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        # self._u = c_puct * np.sqrt((self._P) / (1 + self._n_visits))
        # self._u = c_puct * (self._P) / (1 + self._n_visits)

        # print("Q value: ",self._Q,"   ","U value:  ",self._u)
        # print(("Q value:{}  "
        #       "U value:{}  ").format(self._Q,self._u))

        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):

    def __init__(self, env, policy_value_fn, c_puct=5, n_playout=10000):

        self.env = env
        self._root = TreeNode(None, 0.0)
        self._root._ob = self.env.reset()

        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

        # 在根节点展开
        action_probs, leaf_value = self._policy_value_fn(self.env.acts, self._root._ob.reshape(-1, self.env.ob_dim))

        if not isinstance(leaf_value, np.float32):
            print(leaf_value.type)
            raise ValueError("leaf_value的类型不对")

        a_p = list(copy.deepcopy(action_probs))

        self._root.expand(action_probs)


        # print("action_probs:",action_probs)

        # print(("初始化时，树的根节点中的信息："
        #        "ob:{}  "
        #        "child01:{}  "
        #        "child:{}  "
        #        "child03:{}  "
        #        "child01:{}  ").format(self._root._ob,
        #                               a_p[0],
        #                               a_p[1],
        #                               a_p[2],
        #                               a_p[3]
        #                               ))
        # print(" ")

    def _playout(self):
        """"""
        c_node = self._root
        action = None

        while (1):
            if c_node.is_leaf():
                break
            else:
                action, c_node = c_node.select(self._c_puct)

        self.env.recover(c_node._parent._ob)
        ob, reward, done, _ = self.env.step(action)

        # print("_playout；真实reward：",reward)

        if done == False:
            c_node._ob = ob
            action_probs, leaf_value = self._policy_value_fn(self.env.acts, ob.reshape(-1, self.env.ob_dim))

            # print(leaf_value)
            if not isinstance(leaf_value,np.float32):
                print(leaf_value.type)
                raise ValueError("leaf_value的类型不对")

            c_node.expand(action_probs)
            c_node.update_recursive(leaf_value + reward)
        else:
            c_node.update_recursive(reward)

    def get_move_probs(self, temp=1e-3):
        """

        :param env:
        :param reset_info:
        :param temp:
        :return:
        """

        for n in range(self._n_playout):
            self._playout()

        """根据访问的次数，计算移动的概率"""
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)

        # 调试
        q_u_value = [(node._Q, node._u)
                      for act, node in self._root._children.items()]
        q,u = zip(*q_u_value)

        # print("当前根节点的每个孩子节点的Q值：", q)
        # print("当前根节点的每个孩子节点的U值：", u)

        act_probs = softmax(np.log(np.array(visits) + 1e-10) / temp)

        return acts, act_probs, self._root._Q, self._root._ob

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        """在树中前进，保留我们已经知道的关于子树的一切。"""

        if last_move in self._root._children:
            self._root = self._root._children[last_move]

            if (self._root._ob is None and self._root.is_leaf()):  # self._root._ob==None and
                # print("是新的根节点，但是没状态,且没有孩子")
                self.env.recover(self._root._parent._ob)
                ob, reward, done, _ = self.env.step(last_move)
                self._root._ob = ob

            #测试
            # print("往前走了一步")
            # print(self._root._parent._ob[0])
            # print(last_move, np.array([5, 4, 0]) - self._root._ob[0])

            self._root._parent = None

        else:

            "一条轨迹结束，树重新开始"
            self._root = TreeNode(None, 0.0)
            self._root._ob = self.env.reset()
        # 在根节点展开
        action_probs, leaf_value = self._policy_value_fn(self.env.acts, self._root._ob.reshape(-1, self.env.ob_dim))
        self._root.expand(action_probs)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """
    """

    def __init__(self,
                 env,
                 policy_value_function,
                 c_puct=5,
                 n_playout=20,
                 is_selfplay=0):
        self.policy_value_fn = policy_value_function

        print("define MCTS player")
        self.mcts = MCTS(env, policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        # 环境初始化
        self.mcts.update_with_move(-1)

    def get_action(self, temp=1e-3, Dirichlet_coef=0.1):

        """根据访问次数，返回从根节点的每一个动作的值"""
        acts, probs, v_value, rootOb = self.mcts.get_move_probs(temp=temp)

        # print("从根节点往下选择一步的概率：", probs)

        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for self-play training)
            """
            np.random.choice(a, size=None, replace=True, p=None)
            从一维array a 或 int 数字a 中，以概率p随机选取大小为size的数据，
            replace表示是否重用元素，即抽取出来的数据是否放回原数组中，默认为true（抽取出来的数据有重复）
            """
            """这里size为None,值选出一个动作"""
            """意思就是会根据概率选出动作，动作对应的概率越大，就越可能被选到"""

            action = np.random.choice(
                acts,
                p=(1.0 - Dirichlet_coef) * probs + Dirichlet_coef * np.random.dirichlet(0.3 * np.ones(len(probs)))
            )
            """从Dirichlet狄利克雷分布中抽取样本"""
            """np.random.dirichlet(0.3 * np.ones(5)) 返回的序列的和为1"""
            # update the root node and reuse the search tree

            self.mcts.update_with_move(action)
        else:
            # with the default temp=1e-3, it is almost equivalent等价的
            # to choosing the move with the highest prob
            action = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)

        return action, probs, v_value, rootOb

    # def __str__(self):
    #     return "MCTS {}".format(self.player)
