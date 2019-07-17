# random_map.py

import numpy as np

from Env import point


class RandomMap:
    def __init__(self, size=50):
        self.size = size
        self.obstacle = size // 8  # 6
        self.GenerateObstacle()

    def GenerateObstacle(self):
        self.obstacle_point = []
        # 取整除 - 返回商的整数部分（向下取整）
        self.obstacle_point.append(point.Point(self.size // 2, self.size // 2))  # （25，25）
        self.obstacle_point.append(point.Point(self.size // 2, self.size // 2 - 1))  # （25，24）

        # Generate an obstacle in the middle
        for i in range(self.size // 2 - 4, self.size // 2):  # (21,25)
            self.obstacle_point.append(point.Point(i, self.size - i))
            self.obstacle_point.append(point.Point(i, self.size - i - 1))
            self.obstacle_point.append(point.Point(self.size - i, i))
            self.obstacle_point.append(point.Point(self.size - i, i - 1))

        #固定的障碍物
        xys=[]
        xys.append([20, 15])
        xys.append([35,20])
        xys.append([14, 8])
        xys.append([30, 35])
        xys.append([15, 20])

        count = 0
        for xy in xys:
            x = xy[0]
            y = xy[1]
            self.obstacle_point.append(point.Point(x, y))

            if(count>len(xys)//2):
                for l in range(self.size // 4):  # 12
                    self.obstacle_point.append(point.Point(x, y + l))
                    pass
            else:
                for l in range(self.size // 4):
                    self.obstacle_point.append(point.Point(x + l, y))
                    pass
            count+=1


        #每次随机的障碍物
        # 选出5个点，以0.5的概率向纵向或横向展开12个
        # for i in range(self.obstacle - 1):  # 0、1、2、3、4
        #     x = np.random.randint(0, self.size)
        #     y = np.random.randint(0, self.size)
        #
        #     # print(x,y)
        #
        #     self.obstacle_point.append(point.Point(x, y))
        #
        #     if (np.random.rand() > 0.5):  # Random boolean
        #         for l in range(self.size // 4):  # 12
        #             self.obstacle_point.append(point.Point(x, y + l))
        #             pass
        #     else:
        #         for l in range(self.size // 4):
        #             self.obstacle_point.append(point.Point(x + l, y))
        #             pass

    def IsObstacle(self, i, j):
        for p in self.obstacle_point:
            if i == p.x and j == p.y:
                return True
        return False

    def IsOutside(self, i, j):
        if i < 0 or i >= 50 or j < 0 or j >= 50:
            return True
        return False
