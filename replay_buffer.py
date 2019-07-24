# -*- coding:utf-8 -*-
"""
ddpg中的replay buffer用来存储采样数据
@author: Weijie Shen
"""
from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        """
        定义一个replay buffer类
        :param buffer_size:
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def get_batch(self, batch_size):
        """从buffer中随机采样一个batch的数据"""
        return random.sample(self.buffer, batch_size)

    def size(self):
        """返回当前的buffer大小"""
        return len(self.buffer)

    # def add(self, state, action, reward, new_state, done):
    #     """往buffer中添加数据"""
    #     experience = (state, action, reward, new_state, done)
    #     self.buffer.append(experience)

    def add(self, data):
        """往buffer中添加数据"""
        # experience = (state, action, reward, new_state, done)
        self.buffer.extend(data)

    def erase(self):
        """清空buffer"""
        self.buffer.clear()

if __name__ == "__main__":
    """测试"""
    buffer = ReplayBuffer(5)
    for i in range(6):
        buffer.add(i,i+10,i+20,i+30,i+40)
        print(buffer.size())
