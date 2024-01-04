import os
import json
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.agent_env import SingleAgentEnv
# https://medium.com/nerd-for-tech/reinforcement-learning-introduction-to-policy-gradients-aa2ff134c1b

def reinforce(env: SingleAgentEnv,
              gamma: float = 0.99999,
              lr: float = 0.001,
              max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    model = init_model(env.state_size, env.action_size, lr)

    Returns = {}
    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0

        R =[]
        log_probs = []

        env.reset()
        while not env.is_game_over():
            s = env.state_vector()
            mask = env.available_actions_mask()

            a, lp = model.predict(s.reshape(1, len(s)), verbose=0)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            R.append(r)
            log_probs.append(lp)
            lenght_episode += 1

        G = 0
        for t in reversed(range(lenght_episode)):
            r_t = R[t]
            lp_t = log_probs[t]

            G = r_t + gamma * G
            #train
            # ...

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

        # save logs
        dict_logs = {
            "lenght_episodes": lenght_episodes,
            "reward_episodes": reward_episodes
        }
        logs_path = os.path.join('logs', env.__class__.__name__, 'reinforce')
        logs_name = 'logs.json'
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        with open(os.path.join(logs_path, logs_name), 'w') as file:
            json.dump(dict_logs, file)

        model_save_path = 'model/REINFORCE/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model.save(model_save_path)


def init_model(input_size, output_size, lr):
    model = Sequential([
        Dense(units=64, input_shape=(input_size,), bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu'),
        Dense(units=32, bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu'),
        #Dense(units=output_size, bias_initializer=tf.keras.initializers.RandomNormal(), activation='linear')
        # custom soft max
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return model