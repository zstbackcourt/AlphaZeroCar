# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from mcts.utils import lkrelu,fc
from mcts.logger import MyLogger
# from utils import lkrelu,fc
# from logger import MyLogger


class ddpg(object):
    def __init__(self,
                 ob_dim,
                 nbatch,
                 save_path,
                 batch_size,
                 gamma=0.99,
                 a_lr=0.0001,
                 c_lr=0.0001,
                 tau=0.001,
                 ):
        self.sess=tf.Session()
        self.mylogger = MyLogger("./DDPGlog/logs/")
        self.gamma=gamma
        self.batch_size=batch_size
        self.ob_dim = ob_dim
        self.save_path=save_path
        self.actor = ActorNetwork(state_size=self.ob_dim, action_size=2, lr=a_lr, tau=tau)
        self.critic = CriticNetwork(state_size=self.ob_dim, action_size=2, lr=c_lr, tau=tau)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.mylogger.add_sess_graph(self.sess.graph)
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(max_to_keep=1)
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("ddpg Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("ddpg Could not find saved model")


    def learn(self, mini_batch):#buffer, batch_size):
        #num_steps = int(buffer.size() // batch_size)
        num_steps=1
        for step in range(num_steps):
            #mini_batch = buffer.get_batch(batch_size=batch_size)
            state_batch = [data[0] for data in mini_batch]
            action_batch=[data[3]for data in mini_batch]
            nextstate_batch=[data[4]for data in mini_batch]
            # print(nextstate_batch.shape)
            reward_batch=[data[5]for data in mini_batch]
            done_batch=[data[6]for data in mini_batch]
            #print (action_batch)
            state_batch = np.array(state_batch).reshape(self.batch_size, self.ob_dim - 7)
            action_batch=np.array(action_batch).reshape(self.batch_size, 2)
            nextstate_batch=np.array(nextstate_batch).reshape(self.batch_size, self.ob_dim - 7)
            #print(nextstate_batch.shape)
            reward_batch=np.array(reward_batch).reshape(self.batch_size, 1)
            done_batch=np.array(done_batch).reshape(self.batch_size, 1)

            next_a_target = self.actor.get_action_target(nextstate_batch, self.sess)
            next_q_target = self.critic.get_qvalue_target(nextstate_batch, next_a_target, self.sess)
            # print ("###################reward#######")
            # print (reward_batch)
            # print ("##################q################")
            # print (next_q_target)
            # print ("################done############")
            # print (done_batch[0].astype(int))
            #y=np.zeros(len(mini_batch),dtype=float)
            #for i in mini_batch:
                #y[i]=reward_batch[i] + self.gamma * next_q_target[i] * (1 - done_batch[i].astype(int))
            y = np.array([reward_batch[i] + self.gamma * next_q_target[i] * (1 - done_batch[i]) for i in range(len(mini_batch))])
            y = y.reshape([len(mini_batch)])

            # update ciritc by minimizing l2 loss
            l = self.critic.train(state_batch, action_batch, y, self.sess)
            global_step = self.sess.run(self.global_step)
            if step%10 == 0 and step>0:
                self.mylogger.write_summary_scalar(global_step, "loss", l)
            # update actor policy with sampled gradient
            cur_a_pred = self.actor.get_action(state_batch, self.sess)

            a_gradients = self.critic.get_gradients(state_batch, cur_a_pred, self.sess)
            self.actor.train(state_batch, a_gradients[0], self.sess)

            # update target network:
            self.actor.update_target(self.sess)
            self.critic.update_target(self.sess)
            return l

    def save_model(self):
        print("ddpg 保存模型")
        self.saver.save(self.sess, self.save_path)




class CriticNetwork(object):

    def __init__(self, state_size, action_size, lr, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.tau = tau
        self.input_s, self.action, self.critic_variables, self.q_value = self._build_network("critic")
        self.input_s_target, self.action_target, self.critic_variables_target, self.q_value_target = self._build_network("critic_target")
        self.target = tf.placeholder(tf.float32, [None],name="target")
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_variables])
        self.loss = tf.reduce_mean(tf.square(self.target - self.q_value)) + 0.01 * self.l2_loss
        self.optimize = self.optimizer.minimize(self.loss)
        self.update_target_op = [self.critic_variables_target[i].assign(tf.multiply(self.critic_variables[i], self.tau) + tf.multiply(self.critic_variables_target[i],1 - self.tau)) for i in range(len(self.critic_variables))]
        self.action_gradients = tf.gradients(self.q_value, self.action)

    def _build_network(self, name):
        input_s = tf.placeholder(tf.float32, [None, self.state_size-7],name="input_s")
        action = tf.placeholder(tf.float32, [None, self.action_size],name="action")
        with tf.variable_scope(name):
            p1 = lkrelu(fc(tf.concat((input_s, action), 1), 'q_fc1', nh=256, init_scale=np.sqrt(2.0)))
            p2 = lkrelu(fc(p1, 'q_fc2', nh=128, init_scale=np.sqrt(2.0)))
            p3 = lkrelu(fc(p2, 'q_fc3', nh=64, init_scale=np.sqrt(2.0)))
            q_value = fc(p3, 'q', nh=1, init_scale=np.sqrt(2.0))
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, action, critic_variables, tf.squeeze(q_value)

    def get_qvalue_target(self, state, action, sess):
        return sess.run(self.q_value_target, feed_dict={
            self.input_s_target: state,
            self.action_target: action
        })

    def get_gradients(self, state, action, sess):
        return sess.run(self.action_gradients, feed_dict={
            self.input_s: state,
            self.action: action
        })

    def train(self, state, action, target, sess):
        _, loss = sess.run([self.optimize, self.loss], feed_dict={
            self.input_s: state,
            self.action: action,
            self.target: target
        })
        return loss

    def update_target(self, sess):
        sess.run(self.update_target_op)

            
class ActorNetwork(object):


  def __init__(self, state_size, action_size, lr, tau=0.001):
    self.state_size = state_size
    self.action_size = action_size
    self.optimizer = tf.train.AdamOptimizer(lr)
    self.tau = tau

    self.input_s, self.actor_variables, self.action_values = self._build_network("actor")
    self.input_s_target, self.actor_variables_target, self.action_values_target = self._build_network("actor_target")

    self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
    self.actor_gradients = tf.gradients(self.action_values, self.actor_variables, -self.action_gradients)
    self.update_target_op = [self.actor_variables_target[i].assign(tf.multiply(self.actor_variables[i], self.tau) + tf.multiply(self.actor_variables_target[i], 1 - self.tau))
                              for i in range(len(self.actor_variables))]
    self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_variables))


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.state_size-7])
    with tf.variable_scope(name):
        p1 = lkrelu(fc(input_s, 'pi_fc1', nh=256, init_scale=np.sqrt(2.0)))
        p2 = lkrelu(fc(p1, 'pi_fc2', nh=128, init_scale=np.sqrt(2.0)))
        p3 = lkrelu(fc(p2, 'pi_fc3', nh=64, init_scale=np.sqrt(2.0)))
        # pi = tf.nn.log_softmax(fc(p3, 'pi', nh=self.action_size, init_scale=np.sqrt(2.0)))
        # action_values = tf.tanh(fc(p3, 'action', nh=self.action_size, init_scale=np.sqrt(2.0)))
        action_values = fc(p3, 'action', nh=self.action_size, init_scale=np.sqrt(2.0))
        # action_values = tf.argmax(tf.exp(pi),axis=1)
    actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, actor_variables, action_values


  def get_action(self, state, sess):
    return sess.run(self.action_values, feed_dict={self.input_s: state})


  def get_action_target(self, state, sess):
    return sess.run(self.action_values_target, feed_dict={self.input_s_target: state})


  def train(self, state, action_gradients, sess):
    sess.run(self.optimize, feed_dict={
        self.input_s: state,
        self.action_gradients: action_gradients
      })


  def update_target(self, sess):
    sess.run(self.update_target_op)