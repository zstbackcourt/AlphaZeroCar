# main.py

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Env import random_map

class figure_print:
    def __init__(self,map):
        plt.ion()
        plt.figure(figsize=(5, 5))
        self.map = map

        self.ax = plt.gca() #当前figure的axes坐标轴
        self.ax.set_xlim([0, map.size])
        self.ax.set_ylim([0, map.size])

    def show(self):
        for i in range(self.map.size):
            for j in range(self.map.size):
                if self.map.IsObstacle(i,j):
                    rec = Rectangle((i, j), width=1, height=1, color='gray')
                    self.ax.add_patch(rec)
                else:
                    rec = Rectangle((i, j), width=1, height=1, edgecolor='gray', facecolor='w')
                    self.ax.add_patch(rec)

        rec = Rectangle((0, 0), width = 1, height = 1, facecolor='b')
        self.ax.add_patch(rec)

        rec = Rectangle((self.map.size-1, self.map.size-1), width = 1, height = 1, facecolor='r')
        self.ax.add_patch(rec)

        print("asdfa")
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.ioff()
        plt.show()

# a_star = a_star.AStar(map)
# a_star.RunAndSaveImage(ax, plt)

