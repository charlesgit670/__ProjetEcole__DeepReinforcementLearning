import os
import json
import numpy as np
from tqdm import tqdm
# import time

import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam

from src.agent_env import SingleAgentEnv
from src.DRL_algorithm.function_utils import acceptable_softmax_with_mask

class CustomModel(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = Dense(64, input_shape=(input_size,), bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu')
        self.l2 = Dense(32, bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu')
        self.out = Dense(output_size, bias_initializer=tf.keras.initializers.RandomNormal(), activation='linear', dtype=tf.float32)

    def call(self, input_data, mask):
        mask = tf.cast(mask, dtype=tf.float32)
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        x = acceptable_softmax_with_mask(x, mask)
        return x

    @tf.function
    def predict(self, input_data, mask, training=False):
        return self(input_data, mask,  training=training)


@tf.function
def my_train(model, opt, states, masks, actions, discount_rewards, baseline):
    with tf.GradientTape() as tape:
        p = model(states, masks, training=True)

        indices = tf.range(0, tf.shape(p)[0]) * tf.shape(p)[1] + actions
        lp = tf.math.log(tf.gather(tf.reshape(p, [-1]), indices)) # indices correspondant à la valeur de la policy pour l'action effectué
        loss = -tf.reduce_mean((discount_rewards - baseline) * lp)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

def train_model(model, opt, states, masks, actions, discount_rewards, baseline):
    states = np.array(states, dtype='int32')
    masks = np.array(masks, dtype='int32')
    actions = np.array(actions, dtype='int32')
    discount_rewards = np.array(discount_rewards, dtype='float32')
    baseline = np.array(baseline, dtype='float32')

    my_train(model, opt, states, masks, actions, discount_rewards, baseline)

def reinforce_mean_baseline(env: SingleAgentEnv,
              gamma: float = 0.99999,
              lr: float = 0.001,
              max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    model = CustomModel(env.state_size, env.action_size)
    optimizer = Adam(learning_rate=lr)
    # init baseline
    baseline = 0

    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0

        states = []
        masks = []
        rewards = []
        actions = []


        env.reset()
        while not env.is_game_over():
            s = env.state_vector()
            mask = env.available_actions_mask()

            # pi = model(s.reshape(1, len(s)), mask.reshape(1, len(mask)))
            pi = model.predict(s.reshape(1, len(s)), mask.reshape(1, len(mask)))
            pi = np.array(pi).reshape(len(mask))
            assert (abs(np.sum(pi) - 1) < 1e-3)
            a = np.random.choice([i for i in range(env.action_size)], p=pi)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            states.append(s)
            masks.append(mask)
            rewards.append(r)
            actions.append(a)

            lenght_episode += 1

        G = 0
        discount_rewards = []
        for t in reversed(range(len(states))):
            r_t = rewards[t]
            G = r_t + gamma * G
            discount_rewards.append(G)
        discount_rewards.reverse()
        discount_rewards = np.array(discount_rewards)

        baseline += 0.01 * (np.mean(discount_rewards) - baseline)

        train_model(model, optimizer, states, masks, actions, discount_rewards, baseline)

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    logs_path = os.path.join('logs', env.__class__.__name__, 'reinforce_mean_baseline')
    logs_name = 'logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/REINFORCE_BL/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save(model_save_path)
