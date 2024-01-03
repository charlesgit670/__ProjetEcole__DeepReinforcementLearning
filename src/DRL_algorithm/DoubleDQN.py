import os
import json
import numpy as np
from tqdm import tqdm
import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.DRL_algorithm.function_utils import acceptable_softmax_with_mask, timing_decorator, apply_mask
from src.agent_env import SingleAgentEnv

def double_deep_q_learning(env: SingleAgentDeepEnv,
                    gamma: float = 0.99999,
                    lr: float = 0.001,
                    epsilon: float = 0.2,
                    max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    online_q_net = init_model(env.state_size, env.action_size, lr)
    target_q_net = init_model(env.state_size, env.action_size, lr)
    target_q_net.set_weights(online_q_net.get_weights())
    # init buffer that saved all transition in one episode
    buffer = []

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

            # append step to buffer
            buffer.append((s, a, r, s_p, is_game_done, mask))

            # end_time = time.time()
            # print(end_time - start_time, " secondes")

            G += gamma ** lenght_episode * r
            lenght_episode += 1

        # train model
        train_model(online_q_net, target_q_net, buffer, gamma)
        buffer.clear()
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
    logs_path = os.path.join('logs', env.__class__.__name__, 'deep_q_learning')
    logs_name = 'logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/DQN/'
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
def train_model(online_q_net, target_q_net, buffer, gamma):
    s = []
    s_p = []
    r = []
    a = []
    is_game_over = []
    mask = []
    for elem in buffer:
        s.append(elem[0])
        s_p.append(elem[3])
        r.append(elem[2])
        a.append(elem[1])
        is_game_over.append(elem[4])
        mask.append(elem[5])

    s = np.array(s)
    s_p = np.array(s_p)
    r = np.array(r)
    a = np.array(a)
    is_game_over = np.array(is_game_over)
    mask = np.array(mask)

    Q_s_online = online_q_net.predict(s, verbose=0)
    Q_s_p_online = online_q_net.predict(s_p, verbose=0)
    Q_s_p_target = target_q_net.predict(s_p, verbose=0)
    best_a = np.argmax(apply_mask(Q_s_p_online, mask), axis=1)
    ind = np.array([i for i in range(len(buffer))])
    y_tmp = r + gamma * Q_s_p_target[ind, best_a] * (1 - is_game_over)
    y = Q_s_online.copy()
    y[ind, a.astype(int)] = y_tmp

    online_q_net.fit(x=s, y=y, epochs=1, verbose=0)



