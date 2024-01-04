import os
import json
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.DRL_algorithm.function_utils import apply_mask
from src.agent_env import SingleAgentEnv



"""
Class SumTree et Memory proviennent du git :
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
"""
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # Returns the root node

class Memory(object):  # stored as ( s, a, r, s_, done ) in SumTree

    def __init__(self, capacity, alpha, beta, beta_increment_per_sampling):
        self.tree = SumTree(capacity)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = alpha  # [0~1] convert the importance of TD error to priority
        self.beta = beta  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = 1.  # clipped abs error

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 6), dtype=object), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors = abs(abs_errors) + self.epsilon # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

def init_model(input_size, output_size, lr):
    model = Sequential([
        Dense(units=64, input_shape=(input_size,), bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu'),
        Dense(units=32, bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu'),
        Dense(units=output_size, bias_initializer=tf.keras.initializers.RandomNormal(), activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return model

def init_experience_replay(env, memory):
    replay_memory_size = memory.tree.capacity
    i = 0
    while i < replay_memory_size:
        env.reset()
        while not env.is_game_over() and i < replay_memory_size:
            s = env.state_vector()
            aa = env.available_actions_ids()
            a = np.random.choice(aa)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_vector()
            is_game_done = env.is_game_over()
            mask = env.available_actions_mask()

            transition = np.array((s, a, r, s_p, is_game_done, mask), dtype=object)


            memory.store(transition)
            i += 1


# @timing_decorator
def train_model(online_q_net, target_q_net, replay_buffer, batch_size, gamma):
    # get batch samples
    b_idx, batch_samples, ISWeights = replay_buffer.sample(batch_size)

    Q_s_online = online_q_net.predict(np.vstack(batch_samples[:, 0]), verbose=0)
    Q_s_p_online = online_q_net.predict(np.vstack(batch_samples[:, 3]), verbose=0)
    Q_s_p_target = target_q_net.predict(np.vstack(batch_samples[:, 3]), verbose=0)
    best_a = np.argmax(apply_mask(Q_s_p_online, np.vstack(batch_samples[:, 5])), axis=1)

    ind = np.array([i for i in range(batch_size)])
    y_tmp = batch_samples[:, 2] + gamma * Q_s_p_target[ind, best_a] * (1 - batch_samples[:, 4])
    y = Q_s_online.copy()
    y[ind, batch_samples[:, 1].astype(int)] = y_tmp

    online_q_net.fit(x=np.vstack(batch_samples[:, 0]), y=y, sample_weight=ISWeights, epochs=1, verbose=0)
    td_errors = np.sum(y - Q_s_online, axis=1)
    replay_buffer.batch_update(b_idx, td_errors)


def double_deep_q_learning_with_prioritized_experience_replay(env: SingleAgentEnv,
                    gamma: float = 0.99999,
                    lr: float = 0.001,
                    epsilon: float = 0.2,
                    alpha: float = 0.6,
                    beta: float = 0.4,
                    beta_increment_per_sampling: float = 0.001,
                    replay_memory_size: int = 100,
                    batch_size: int = 32,
                    max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    online_q_net = init_model(env.state_size, env.action_size, lr)
    target_q_net = init_model(env.state_size, env.action_size, lr)
    target_q_net.set_weights(online_q_net.get_weights())
    # init prioritized replay experience
    replay_buffer = Memory(replay_memory_size, alpha=alpha, beta=beta, beta_increment_per_sampling=beta_increment_per_sampling)
    init_experience_replay(env, replay_buffer)

    for ep in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0
        env.reset()
        while not env.is_game_over():
            # start_time = time.time()
            s = env.state_vector()
            aa = env.available_actions_ids()
            mask = env.available_actions_mask()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                Q_s = online_q_net.predict(s.reshape(1, len(s)), verbose=0)
                a = np.argmax(apply_mask(Q_s, mask[None, :]))

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_vector()
            is_game_done = env.is_game_over()
            mask = env.available_actions_mask()

            # update replay buffer
            replay_buffer.store(np.array((s, a, r, s_p, is_game_done, mask), dtype=object))

            # end_time = time.time()
            # print(end_time - start_time, " secondes")

            G += gamma ** lenght_episode * r
            lenght_episode += 1

        # train online model
        train_model(online_q_net, target_q_net, replay_buffer, batch_size, gamma)
        # update target q net
        if ep % 10 == 0:
            target_q_net.set_weights(online_q_net.get_weights())

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    logs_path = os.path.join('logs', env.__class__.__name__, 'double_deep_q_learning_with_prioritized_experience_replay')
    logs_name = 'logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/DDQN_PER/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    online_q_net.save(model_save_path)