# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import tensorflow as tf
import numpy as np
import gym
from simple_env import SnakeEnv

# env = gym.make('CartPole-v0').unwrapped
env = SnakeEnv(0,False)


saver = tf.train.import_meta_graph("./model/snake.meta")
with tf.Session() as session:
    saver.restore(session, tf.train.latest_checkpoint("./model/"))
    graph = tf.get_default_graph()
    s = env.reset()
    ep_r = 0
    # for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
    #     print(tensor_name)
    while True:
        # env.render()
        s = np.array(s).reshape((-1,3))
        ap = graph.get_tensor_by_name("Global_Net/actor/ap/Softmax:0")
        ap_ = session.run(ap,feed_dict={"Global_Net/S:0":s})
        # print(ap_)
        action = np.argmax(ap_[0])
        s_, r, done, info = env.step(action)
        if done:
            print(done)
            # r = -5
            s = env.reset()
            ep_r += r
            print("reward:{}".format(ep_r))
            ep_r = 0
            s = env.reset()
        else:
            s = s_
            ep_r += r
        # if done: r = -5
        # ep_r += r
        # action = np.random.choice(range(ap_.shape[1]),
        #                           p=ap_.ravel())
        # print(action)