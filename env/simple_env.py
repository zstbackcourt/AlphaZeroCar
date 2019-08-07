# -*- coding:utf-8 -*-
"""
一个用pygame写的简单的env（新版）
@author: Weijie Shen
"""
# pygame游戏库，sys操控python运行的环境
import pygame, sys, random
# 这个模块包含所有pygame所使用的常亮
from pygame.locals import *

import numpy as np
from gym import spaces

# 1,定义颜色变量
# 0-255  0黑色  255白色
redColor = pygame.Color(255, 0, 0)
# 背景为黑色
greenColor = pygame.Color(0, 255, 0)
# 贪吃蛇为白色
whiteColor = pygame.Color(255, 255, 255)

#墙的颜色
blackColor = pygame.Color(0,0,0)


class SnakeEnv:
    def __init__(self,gameSpeed=0, train_model = False):
        """"""
        """游戏相关:
        """
        # 初始化pygame
        pygame.init()

        # 定义一个变量来控制速度
        self.gameSpeed = gameSpeed
        self.train_model = train_model

        self.fpsClock = pygame.time.Clock()

        # 创建pygame显示层，创建一个界面
        if(self.train_model!=True):
            self.playsurface = pygame.display.set_mode((640, 480))
            pygame.display.set_caption('MCTS')

        # 初始化变量
        # 贪吃蛇初始坐标位置   （先以100,100为基准）
        self.snakePosition = [100, 100]

        # 初始化贪吃蛇的长度列表中有几个元素就代表有几段身体
        # 只有一个方块长度
        self.snakeBody = [[100, 100]]

        # 初始化目标的位置
        self.targetPosition = [500, 380]

        # 目标方块的标记 目的：判断是否吃掉了这个目标方块 1 就是没有吃 0就是吃掉
        self.targetflag = 1

        # 初始化方向   --》往右
        self.direction = 'right'

        # 定义一个方向变量（人为控制  按键）
        self.changeDirection = self.direction

        self.direction_ = 0

        """agent相关
        ob：相对坐标、到边缘位置、当前方向 
        """
        self.acts = [0,1,2,3]
        self.actionDic = {0: "right",
                          1: "left",
                          2: "up",
                          3: "down"}

        self.MaxDis = np.sqrt(400**2+280**2)
        self.preDis = self.MaxDis

        self.num_envs = 1
        self.envs = 1

        '''get brain info'''
        self.act_dim = 4
        self.ob_dim = 3

        '''set action/ob space'''
        # self.action_space = spaces.Box(low=np.zeros([self.act_dim]) - 1, high=np.zeros([self.act_dim]) + 1,
        #                                dtype=np.float32)  # 转向、油门、刹车

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.zeros([self.ob_dim]) - 1, high=np.zeros([self.ob_dim]) + 1,
                                            dtype=np.float32)

        self.walls = []
        for i in range(15):
            self.walls.append([300,i*20])
        for i in range(17):
            self.walls.append([300+i*20,300])
        for i in range(7):
            self.walls.append([i*20,200])
        for i in range(12):
            self.walls.append([140,200+i*20])
        for i in range(25):
            self.walls.append([140+i*20,440])
        # self.walls.append([260,280])
        # self.walls.append([280, 260])
        # #self.walls.append([220,180])
        # self.walls.append([180,160])
        # self.walls.append([220, 380])
        # self.walls.append([360, 360])
        # print(self.walls)

    def step(self, action):
        """"""
        r = 0.0
        d = [False]
        infos = {}


        # random_var = np.random.rand()
        # if random_var <0.25:
        #     a = np.random.randint(0, 4)

        """执行动作"""
        self.changeDirection = self.actionDic[action]
        #print("执行action前的snakePosition{}".format(self.snakePosition))
        self.execute_action()
        # if self.snakePosition[0] > 240 and self.snakePosition[1] < 360:
        #     print("在墙里")
        #print("执行action后的snakePostion{}".format(self.snakePosition))
        # print("self.snakePosition:",self.snakePosition)
        # 执行完动作之后，判断游戏是否结束
        if self.IsGameover():  # 结束了
            # gameover()
            # if self.snakePosition[0] >240 and self.snakePosition[1] <360:
            #     gameover()
            ob = self.reset()
            r = r - 10.0
            d = [True]

        else:
            # 增加蛇的长度
            self.snakeBody.insert(0, list(self.snakePosition))
            # 如果贪吃蛇和目标方块的位置重合
            if self.snakePosition[0] == self.targetPosition[0] and self.snakePosition[1] == self.targetPosition[1]:
                print("到达目标点")
                self.targetflag = 0
                self.snakeBody.pop()
                r = r + 10
                ob = self.reset()
                d = [True]
            else:
                relative_pos = np.array(self.targetPosition) - np.array(self.snakePosition)
                # x_edge = [self.snakePosition[0] - 0, 620 - self.snakePosition[0]]
                # y_edge = [self.snakePosition[1] - 0, 460 - self.snakePosition[1]]
                # ob = [relative_pos[0], relative_pos[1], x_edge[0], x_edge[1], y_edge[0], y_edge[1], self.direction_]

                ob = [relative_pos[0]/100.0, relative_pos[1]/100.0, self.direction_//100]

                self.snakeBody.pop()
                currentDis = np.sqrt((self.targetPosition[0] - self.snakePosition[0]) ** 2 + (
                        self.targetPosition[1] - self.snakePosition[1]) ** 2)
                if self.preDis - currentDis <=0:
                    approachR = (((self.preDis - currentDis) / (self.MaxDis + 1e-8))) * 5.0 *1.2
                else:
                    approachR = (((self.preDis - currentDis) / (self.MaxDis + 1e-8))) * 5.0
                r += approachR
                self.preDis = currentDis

        # 填充背景颜色

        if (self.train_model!=True):

            self.playsurface.fill(whiteColor)

            for position in self.snakeBody:
                # 第一个参数serface指定一个serface编辑区，在这个区域内绘制
                # 第二个参数color：颜色
                # 第三个参数:rect:返回一个矩形(xy),(width,height)
                # 第四个参数：width：表示线条的粗细  width0填充  实心
                # 化蛇
                # print("adfadf")
                pygame.draw.rect(self.playsurface, redColor, Rect(position[0], position[1], 20, 20))
                pygame.draw.rect(self.playsurface, greenColor, Rect(self.targetPosition[0], self.targetPosition[1], 20, 20))
                for wall in self.walls:
                    pygame.draw.rect(self.playsurface, blackColor, Rect(wall[0], wall[1], 20, 20))
                    pass
            pygame.event.get()
            pygame.display.flip()

        # 控制游戏速度
        # if self.gameSpeed==0
        self.fpsClock.tick(self.gameSpeed)


        # print(ob)
        ob = np.array(ob).reshape(-1,self.ob_dim)

        #print("step 输出的ob{}和当前的snakePostion{}".format(ob,self.snakePosition))
        return ob, r, np.array(d), infos

    def recover(self,ob):
        #print("recover 接收的ob{},当前的snakePostion{}".format(ob,self.snakePosition))
        ob = [round(ob[0][0]*100),round(ob[0][1]*100),round(ob[0][2]*100)]
        self.snakePosition = [self.targetPosition[0]-ob[0], self.targetPosition[1]-ob[1]]
        self.preDis = np.sqrt((self.targetPosition[0] - self.snakePosition[0]) ** 2 + (
                    self.targetPosition[1] - self.snakePosition[1]) ** 2)
        #print("recover 恢复后的snakePostion{}".format(self.snakePosition))
        self.direction_ = ob[2]

        if(self.direction_ == 0):
            self.direction = "right"
            # print("")
        elif(self.direction_==100):
            self.direction = "left"
        elif (self.direction_ == 200):
            self.direction = "up"
        elif (self.direction_ == 300):
            self.direction = "down"

    def reset(self):

        """
        :return:
        """
        if self.targetflag == 0:
            x = random.randrange(1, 31)
            y = random.randrange(1, 23)
            self.targetPosition = [int(x * 20), int(y * 20)]
            self.targetflag = 1

        self.targetPosition = [500, 380]

        self.snakePosition = [100, 100]
        self.snakeBody = [[100, 100]]

        # self.MaxDis = np.sqrt((self.targetPosition[0] - self.snakePosition[0]) ** 2 + (
        #         self.targetPosition[1] - self.snakePosition[1]) ** 2)
        self.MaxDis = np.sqrt(400**2+280**2)
        self.preDis = self.MaxDis

        self.direction = 'right'
        self.changeDirection = self.direction
        self.direction_ = 0

        relative_pos = np.array(self.targetPosition) - np.array(self.snakePosition)
        # x_edge = [self.snakePosition[0] - 0, 620 - self.snakePosition[0]]
        # y_edge = [self.snakePosition[1] - 0, 460 - self.snakePosition[1]]
        # ob = [relative_pos[0], relative_pos[1], x_edge[0], x_edge[1], y_edge[0], y_edge[1], self.direction_]
        ob = [relative_pos[0]/100.0, relative_pos[1]/100.0, self.direction_//100]

        ob = np.array(ob).reshape(-1,self.ob_dim)
        #print("reset 输出的ob:{},当前的snakePosition:{}".format(ob,self.snakePosition))
        # print("reset 输出的ob",(ob//100),self.snakePosition)
        return ob

    def execute_action(self):

        # 确定方向
        if self.changeDirection == 'left' and not self.direction == 'right':
            self.direction = self.changeDirection
        if self.changeDirection == 'right' and not self.direction == 'left':
            self.direction = self.changeDirection
        if self.changeDirection == 'up' and not self.direction == 'down':
            self.direction = self.changeDirection
        if self.changeDirection == 'down' and not self.direction == 'up':
            self.direction = self.changeDirection
        # if self.changeDirection == 'left' :
        #     self.direction = self.changeDirection
        # if self.changeDirection == 'right' :
        #     self.direction = self.changeDirection
        # if self.changeDirection == 'up' :
        #     self.direction = self.changeDirection
        # if self.changeDirection == 'down' :
        #     self.direction = self.changeDirection



        # 根据方向移动蛇头
        if self.direction == 'right':
            self.snakePosition[0] += 20
            self.direction_ = 0

        if self.direction == 'left':
            self.snakePosition[0] -= 20
            self.direction_ = 100

        if self.direction == 'up':
            self.snakePosition[1] -= 20
            self.direction_ = 200

        if self.direction == 'down':
            self.snakePosition[1] += 20
            self.direction_ = 300

    def IsGameover(self):
        # 判断是否游戏结束
        if self.snakePosition[0] > 620 or self.snakePosition[0] < 0:
            return True
        elif self.snakePosition[1] > 460 or self.snakePosition[1] < 0:
            return True
        elif self.snakePosition in self.walls:
        #elif self.snakePosition[0] >=240 and self.snakePosition[1] <=360:
            # if self.snakePosition in self.walls:
            #     print("在墙上",self.snakePosition)
            # else:
            #     print("在墙里",self.snakePosition)
            return True
        return False


# 定义游戏结束的函数
def gameover():
    pygame.quit()
    sys.exit()

# a = snake()
# for i in range(1000):
#     a.step([0,1])

