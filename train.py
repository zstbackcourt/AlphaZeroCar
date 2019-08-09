# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from simple_env import SnakeEnv
from ac_net import ACNet

GLOBAL_NET_SCOPE = 'Global_Net'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 3500
UPDATE_GLOBAL_ITER = 20
GAMMA = 0.9
N_S = 3
N_A = 4
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

# def __init__(self, sess,N_S,N_A,scope,lra,lrc,globalAC=None,entropy_beat = 0.001):


class Worker(object):
    def __init__(self, sess,name, globalAC):
        # self.env = gym.make(GAME).unwrapped
        self.env = SnakeEnv(0,True)
        self.name = name
        self.AC = ACNet(sess,N_S,N_A,name,LR_A,LR_C,globalAC) # local net

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                # if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        s_ = np.array(s_).reshape((-1,N_S))
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_})[0, 0]
                        # v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(SESS,N_S,N_A,GLOBAL_NET_SCOPE,LR_A,LR_C)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(SESS,i_name,GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    GLOBAL_AC.saver.save(SESS,"model/snake")
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()