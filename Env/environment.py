# a_star.py

import sys
import time

import numpy as np

from matplotlib.patches import Rectangle

from Env import point
from gym import error, spaces

from Env import random_map
from Env.figure import figure_print

class Env:
    def __init__(self):

        self.map = random_map.RandomMap()

        self.agentP = point.Point(0, 0)


        self.MaxDis = np.sqrt(2 * 49 ** 2)
        self.preDis = self.MaxDis

        # 终点
        self.endP = point.Point(49, 49)

        self.num_envs = 1
        self.envs = 1

        '''get brain info'''
        self.act_dim = 4
        self.ob_dim = 2

        '''set action/ob space'''
        self.action_space = spaces.Box(low=np.zeros([self.act_dim]) - 1, high=np.zeros([self.act_dim]) + 1,
                                       dtype=np.float32)  # 转向、油门、刹车
        self.observation_space = spaces.Box(low=np.zeros([self.ob_dim]) - 1, high=np.zeros([self.ob_dim]) + 1,
                                            dtype=np.float32)

        self.acts = [1, 2, 3, 4]

        self.actionDic = {0: [0, 1],
                       1: [0, -1],
                       2: [-1, 0],
                       3: [1, 0]}

        self.fp = figure_print(self.map)

    def step(self, action):
        """
        action为4维，上1000 下0100 左0010 右0001
        :param action:
        :return:
        """
        """
        r = 0
        d = false
        
        执行动作：
            if是有效点：
                if是终点：
                     r+=到达目标奖励
                     d = True
                     环境reset；
            else:
                碰撞惩罚；
                环境reset；
                d = True;   
                 
            reward+=接近奖励；    
            返回 状态、奖励、d、infos
            
        """
        ob = None
        r = 0.0
        d = [False]
        infos = {}

        # 执行动作
        # print("action probs:",np.exp(action[0]))


        a = np.argmax(np.exp(action[0]), 0)
        # print("action index:", a)


        trueAction = self.actionDic[a]
        self.agentP.x += trueAction[0]
        self.agentP.y += trueAction[1]
        # 新的坐标，上一帧到目标的距离（在恢复时，需要用到）
        # ob = [self.agentP.x, self.agentP.y, self.preDis]
        ob = np.array([self.agentP.x, self.agentP.y]).reshape(-1,2)

        # 判断是否为有效点
        if self.IsValidPoint(self.agentP):
            # 接近奖励
            currentDis = np.sqrt((self.endP.x - self.agentP.x) ** 2 + (self.endP.y - self.agentP.y) ** 2)
            approachR = ((self.preDis - currentDis) / self.MaxDis)*5.0
            r += approachR
            self.preDis = currentDis

            # 判断是否为终点
            if self.IsEndPoint(self.agentP):
                r += 10
                d = [True]
                ob = self.reset()
        else:
            r = r - 10
            d = [True]
            ob = self.reset()

        self.fp.show()

        return ob, r, np.array(d), infos

    def reset(self):
        """
        重置环境，重新开始
        :return:
        """
        self.agentP.x = 0
        self.agentP.y = 0
        self.preDis = self.MaxDis
        # return np.array([self.agentP.x, self.agentP.y, self.preDis]).reshape(-1,3)
        return np.array([self.agentP.x, self.agentP.y]).reshape(-1,2)

    def recoverPos(self, x, y, preDis):
        """
        根据需求，恢复agent的状态
        :param x:
        :param y:
        :return:
        """
        self.agentP.x = x
        self.agentP.y = y
        self.preDis = preDis

    def IsValidPoint(self, p):
        """
        agent出界或是障碍物时返回false
        :param x:
        :param y:
        :return:
        """
        if p.x < 0 or p.y < 0:
            return False
        if p.x >= self.map.size or p.y >= self.map.size:
            return False
        return not self.map.IsObstacle(p.x, p.y)

    def IsStartPoint(self, p):
        """
        判断agent是否为初始点
        :param p:
        :return:
        """
        return p.x == 0 and p.y == 0

    def IsEndPoint(self, p):
        """
        判断agent是否为终点
        :param p:
        :return:
        """
        return p.x == self.map.size - 1 and p.y == self.map.size - 1
