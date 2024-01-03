import os
import json
import numpy as np
from tqdm import tqdm
# import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.DRL_algorithm.function_utils import acceptable_softmax_with_mask, timing_decorator, apply_mask
from src.agent_env import SingleAgentEnv

def double_deep_q_learning_with_experience_replay(env: SingleAgentDeepEnv,
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
    online_q_net = init_model(env.state_size, env.action_size, lr)
    target_q_net = init_model(env.state_size, env.action_size, lr)
    target_q_net.set_weights(online_q_net.get_weights())
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
            replay_buffer[buffer_index] = np.array((s, a, r, s_p, is_game_done, mask), dtype=object)

            buffer_index += 1
            if buffer_index == replay_memory_size:
                buffer_index = 0

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
def init_model(input_size, output_size, lr):
    model = Sequential([
        Dense(units=64, input_shape=(input_size,), bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu'),
        Dense(units=32, bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu'),
        Dense(units=output_size, bias_initializer=tf.keras.initializers.RandomNormal(), activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return model
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

# @timing_decorator
def train_model(online_q_net, target_q_net, replay_buffer, batch_size, gamma):
    # get batch samples
    random_index = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
    batch_samples = replay_buffer[random_index]

    Q_s_online = online_q_net.predict(np.vstack(batch_samples[:, 0]), verbose=0)
    Q_s_p_online = online_q_net.predict(np.vstack(batch_samples[:, 3]), verbose=0)
    Q_s_p_target = target_q_net.predict(np.vstack(batch_samples[:, 3]), verbose=0)
    best_a = np.argmax(apply_mask(Q_s_p_online, np.vstack(batch_samples[:, 5])), axis=1)

    ind = np.array([i for i in range(batch_size)])
    y_tmp = batch_samples[:, 2] + gamma * Q_s_p_target[ind, best_a] * (1 - batch_samples[:, 4])
    y = Q_s_online.copy()
    y[ind, batch_samples[:, 1].astype(int)] = y_tmp

    online_q_net.fit(x=np.vstack(batch_samples[:, 0]), y=y, epochs=1, verbose=0)



