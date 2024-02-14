import os
import json
import numpy as np
from tqdm import tqdm
import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.DRL_algorithm.function_utils import timing_decorator, apply_mask
from src.agent_env import SingleAgentEnv

def deep_q_learning(env: SingleAgentEnv,
                    # Cautious Learner:
                    gamma: float = 0.99,
                    lr: float = 0.0001,
                    epsilon: float = 0.1,

                    # Balanced Strategist:
                    # gamma: float = 0.5,
                    # lr: float = 0.01,
                    # epsilon: float = 0.75,

                    # Bold Explorer:
                    # gamma: float = 0.8,
                    # lr: float = 0.1,
                    # epsilon: float = 0.5,

                    # default values:
                    # gamma: float = 0.99999,
                    # lr: float = 0.001,
                    # epsilon: float = 0.2,

                    max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    model = CustomModel(env.state_size, env.action_size)
    optimizer = Adam(learning_rate=lr)
    # init buffer that saved all transition in one episode
    buffer = []

    for _ in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0
        env.reset()
        while not env.is_game_over():

            s = env.state_vector()
            aa = env.available_actions_ids()
            mask = env.available_actions_mask()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                # Q_s = model(s.reshape(1, len(s)))
                Q_s = model.predict(s.reshape(1, len(s)))
                a = np.argmax(apply_mask(np.array(Q_s), mask[None, :]))

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_vector()
            is_game_done = env.is_game_over()
            mask = env.available_actions_mask()

            # append step to buffer
            buffer.append((s, a, r, s_p, is_game_done, mask))



            G += gamma ** lenght_episode * r
            lenght_episode += 1

        # train model
        train_model(model, optimizer, buffer, gamma)
        buffer.clear()

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    logs_path = os.path.join('logs', env.__class__.__name__, 'deep_q_learning')
    logs_name = 'logs_cautious_learner.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/DQN/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save(model_save_path)


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
def my_train(model, opt, s, y):
    with tf.GradientTape() as tape:
        Q_s = model(s, training=True)
        loss = tf.reduce_mean(tf.square(Q_s - y))

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))


# @timing_decorator
def train_model(model, opt, buffer, gamma):
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

    Q_s = np.array(model.predict(s))
    Q_s_p = np.array(model.predict(s_p))
    y_tmp = r + gamma * np.max(apply_mask(Q_s_p, mask), axis=1) * (1 - is_game_over)
    y = Q_s.copy()
    ind = np.array([i for i in range(len(buffer))])
    # print('y', y)
    # print('ind', ind)
    # print('a nom', a)
    #
    # print(f'len y {len(y)} len ind {len(ind)} len a {len(a)} len y_tmp {len(y_tmp)}')
    #
    # print('a astyupe', a.astype(int))
    y[ind, a.astype(int)] = y_tmp

    my_train(model, opt, s, y)

    # model.fit(x=s, y=y, epochs=1, verbose=0)



