# -*- coding: utf-8 -*-

""" 'pong' game agent player. Uses OpenAI gym"""

import os
import time

import gym
import moviepy.editor as mpy
import numpy as np
import pandas as pd
import tensorflow as tf

n_units = 200
learning_rate = 1e-3
gamma = 0.99
RANDOM_SEED = 2018
RENDER_THRESHOLD = 10
SAVE_INTERVAL = 10
MAX_EPOCH = 3000
MODEL_DIR = './results/'
LOG_FILE = './results/log.csv'
env = gym.make("Pong-v0")
nn_model = './results/save.ckpt'
# nn_model = None


class PongAgent(object):
    def __init__(self, sess):
        self.s_batch = []
        self.a_batch = []
        self.r_batch = []
        self.advantage = []
        self.wins = 0
        self.epoch_number = 0
        self.running_reward = None
        self.cur_x = None
        self.prev_x = None
        self.render = False  # True
        self.s_dim = 80 * 80
        self.statistic = {'reward_sum': [], 'running_reward': []}
        self.sess = sess
        self.observation = env.reset()
        self.build_graph_op()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        if nn_model is not None:
            self.saver.restore(self.sess, nn_model)
            print("model restored")

    def build_net(self):
        with tf.variable_scope('build_net'):
            inputs = tf.placeholder("float", shape=[None, self.s_dim])
            reg = tf.contrib.layers.l2_regularizer(1e-3)
            output = tf.layers.dense(inputs, units=n_units, activation=tf.nn.relu, kernel_regularizer=reg,
                                     kernel_initializer=tf.random_normal_initializer(0., .1), name='h1')
            output = tf.layers.dense(output, units=1, kernel_regularizer=reg,
                                     kernel_initializer=tf.random_normal_initializer(0., .1), name='out')
            output = tf.squeeze(output)
            return inputs, output

    def build_graph_op(self):
        with tf.variable_scope('tf_graph'):
            self.inputs, self.out = self.build_net()
            self.label = tf.placeholder("float", shape=[None])
            self.advantage = tf.placeholder("float", shape=[None])

            with tf.variable_scope('cost'):
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.label)
                self.advantage_loss = tf.multiply(self.loss, self.advantage)

            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.advantage_loss)

    def train(self):
        start = time.time()

        while self.epoch_number < MAX_EPOCH:
            self.play_one_epoch()

            if self.epoch_number % SAVE_INTERVAL == 0:
                self.saver.save(self.sess, MODEL_DIR + "save.ckpt")
                print("model saved.")

        df = pd.DataFrame(self.statistic)
        df.to_csv(LOG_FILE, encoding='utf-8')
        print("Train over, {} epoch total spend {} minutes".format(MAX_EPOCH, round((time.time() - start) / 60, 2)))

    def play_one_epoch(self):
        self.clean_batch()
        while True:
            if self.render: env.render()

            done = self.take_one_step()
            if done:
                self.update_net()
                self.one_epoch_statistic()
                break

    def take_one_step(self):
        self.cur_x = self.preprocess(self.observation)
        state_diff = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.s_dim)
        self.prev_x = self.cur_x
        out = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(state_diff, (1, self.s_dim))})
        up_prob = self.sigmoid(out)
        action = self.get_action(up_prob)
        self.observation, reward, done, info = env.step(action)
        label = self.map_action(action)

        if reward != 0:
            print(('ep %d: game finished, reward: %f' % (self.epoch_number, reward)) + (
                '' if reward == -1 else ' !!!!!!!!'))

        self.s_batch.append(state_diff)
        self.a_batch.append(label)
        self.r_batch.append(reward)
        return done

    def update_net(self):
        self.epoch_number += 1
        discount_r = self.discount_rewards(np.vstack(self.r_batch))
        discount_r = self.standarlize_rewards(discount_r)
        self.sess.run(self.optimizer, feed_dict={self.inputs: np.vstack(self.s_batch), self.label: self.a_batch,
                                                 self.advantage: np.reshape(discount_r, -1)})

    def discount_rewards(self, r):
        """ compute discounted reward """
        discount_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0
            running_add = running_add * gamma + r[t]
            discount_r[t] = running_add
        return discount_r

    def standarlize_rewards(self, r):
        r -= np.mean(r)
        r /= np.std(r)
        return r

    def preprocess(self, state):
        """ prepro 210x160x3 uint8 frame into 6400 (80*80) 1D float vector """
        state = state[35:195]
        state = state[::2, ::2, 0]
        state[state == 144] = 0
        state[state == 109] = 0
        state[state != 0] = 1
        return state.astype(np.float).ravel()

    def clean_batch(self):
        del self.s_batch[:]
        del self.a_batch[:]
        del self.r_batch[:]
        self.observation = env.reset()
        self.cur_x = None
        self.prev_x = None

    def one_epoch_statistic(self):
        reward_sum = np.sum(self.r_batch)
        self.running_reward = reward_sum if self.running_reward is None else self.running_reward * 0.95 + reward_sum * 0.05
        print("resetting evn. episode reward total was %f. running mean: %f." % (reward_sum, self.running_reward))

        self.statistic['reward_sum'].append(reward_sum)
        self.statistic['running_reward'].append(self.running_reward)

        if self.running_reward > RENDER_THRESHOLD:
            self.render = True

    def map_action(self, action):
        return 1 if action == 2 else 0  # fake label

    def get_action(self, up_prob):
        return 2 if np.random.uniform() < up_prob else 3

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def make_frame(self, t):
        new_frame = env.render(mode='rgb_array')
        new_frame[0][:8] = new_frame[0][8]

        self.cur_x = self.preprocess(self.observation)
        state = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.s_dim)
        self.prev_x = self.cur_x

        aprob = self.sigmoid(self.sess.run(self.out, feed_dict={self.inputs: np.reshape(state, (1, self.s_dim))}))
        action = 2 if np.random.uniform() < aprob else 3

        # step the environment
        self.observation, reward, done, info = env.step(action)

        return new_frame


def main():
    with tf.Session() as sess:
        np.random.seed(RANDOM_SEED)

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        agent = PongAgent(sess)
        agent.train()

        agent.observation = env.reset()
        clip = mpy.VideoClip(agent.make_frame, duration=100)
        clip.write_gif("./figures/Pong.gif", fps=15)


if __name__ == '__main__':
    main()
