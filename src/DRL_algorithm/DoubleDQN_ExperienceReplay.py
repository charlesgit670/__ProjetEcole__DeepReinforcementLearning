import os
import json
import numpy as np
from tqdm import tqdm
# import time

import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam

from src.DRL_algorithm.function_utils import timing_decorator, apply_mask
from src.agent_env import SingleAgentEnv

def double_deep_q_learning_with_experience_replay(env: SingleAgentEnv,
                    gamma: float = 0.99999,
                    lr: float = 0.001,
                    epsilon: float = 0.2,
                    replay_memory_size: int = 100,
                    batch_size: int = 32,
                    max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    online_q_net = CustomModel(env.state_size, env.action_size)
    target_q_net = CustomModel(env.state_size, env.action_size)
    dummy_input = tf.constant([[0.0] * env.state_size], dtype=tf.float32)
    # faire un predict à vide permet de provoquer l'iinitialisation des poids
    online_q_net.predict(dummy_input)
    target_q_net.predict(dummy_input)

    target_q_net.set_weights(online_q_net.get_weights())
    optimizer = Adam(learning_rate=lr)
    # init replay experience
    train_frequency_index = 0
    buffer_index = 0
    replay_buffer = init_experience_replay(env, replay_memory_size)

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
                Q_s = online_q_net.predict(s.reshape(1, len(s)))
                a = np.argmax(apply_mask(Q_s, mask[None, :]))

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_vector()
            is_game_done = env.is_game_over()
            mask = env.available_actions_mask()

            # update replay buffer
            replay_buffer[buffer_index] = np.array((s, a, r, s_p, is_game_done, mask), dtype=object)

            buffer_index += 1
            if buffer_index == replay_memory_size:
                buffer_index = 0

            # end_time = time.time()
            # print(end_time - start_time, " secondes")

            G += gamma ** lenght_episode * r
            lenght_episode += 1

        # train online model
        train_model(online_q_net, target_q_net, optimizer, replay_buffer, batch_size, gamma)
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
    logs_path = os.path.join('logs', env.__class__.__name__, 'double_deep_q_learning_with_experience_replay')
    logs_name = 'logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/DDQN_ER/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    online_q_net.save(model_save_path)


# @timing_decorator
def init_experience_replay(env, replay_memory_size):
    replay_buffer = np.zeros((replay_memory_size, 6), dtype=object)
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

            transition = (s, a, r, s_p, is_game_done, mask)


            replay_buffer[i] = np.array(transition, dtype=object)
            i += 1

    return replay_buffer

class CustomModel(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = Dense(64, input_shape=(input_size,), bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu')
        self.l2 = Dense(32, bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu')
        self.out = Dense(output_size, bias_initializer=tf.keras.initializers.RandomNormal(), activation='linear', dtype=tf.float32)

    def call(self, input_data):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        return x

    @tf.function
    def predict(self, input_data, training=False):
        return self(input_data, training=training)

@tf.function
def my_train(online_q_net, opt, s, y):
    with tf.GradientTape() as tape:
        Q_s_online = online_q_net(s, training=True)
        loss = tf.reduce_mean(tf.square(Q_s_online - y))

    grads = tape.gradient(loss, online_q_net.trainable_variables)
    opt.apply_gradients(zip(grads, online_q_net.trainable_variables))

# @timing_decorator
def train_model(online_q_net, target_q_net, opt, replay_buffer, batch_size, gamma):
    # get batch samples
    random_index = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
    batch_samples = replay_buffer[random_index]

    Q_s_online = np.array(online_q_net.predict(np.vstack(batch_samples[:, 0])))
    Q_s_p_online = np.array(online_q_net.predict(np.vstack(batch_samples[:, 3])))
    Q_s_p_target = np.array(target_q_net.predict(np.vstack(batch_samples[:, 3])))
    best_a = np.argmax(apply_mask(Q_s_p_online, np.vstack(batch_samples[:, 5])), axis=1)

    ind = np.array([i for i in range(batch_size)])
    y_tmp = batch_samples[:, 2] + gamma * Q_s_p_target[ind, best_a] * (1 - batch_samples[:, 4])
    y = Q_s_online.copy()
    y[ind, batch_samples[:, 1].astype(int)] = y_tmp

    my_train(online_q_net, opt, np.vstack(batch_samples[:, 0]), y)



