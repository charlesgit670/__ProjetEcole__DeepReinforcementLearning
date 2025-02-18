import os
import json
import numpy as np
from tqdm import tqdm
# import time
from collections import deque

import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam

from src.agent_env import SingleAgentEnv
from src.DRL_algorithm.function_utils import acceptable_softmax_with_mask

class PolicyModel(tf.keras.Model):
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

class ValueModel(tf.keras.Model):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = Dense(64, input_shape=(input_size,), bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu')
        self.l2 = Dense(32, bias_initializer=tf.keras.initializers.RandomNormal(), activation='relu')
        self.out = Dense(1, bias_initializer=tf.keras.initializers.RandomNormal(), activation='linear', dtype=tf.float32)

    def call(self, input_data):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        return x

    @tf.function
    def predict(self, input_data, training=False):
        return self(input_data,  training=training)

@tf.function
def my_train(policy_model, value_model, opt_policy, opt_value, states, masks, actions, deltas, discount_rewards, actions_probs, epsilon, c2):
    with tf.GradientTape() as tape_policy:
        p = policy_model(states, masks, training=True)
        indices = tf.range(0, tf.shape(p)[0]) * tf.shape(p)[1] + actions # indices correspondant à la valeur de la policy pour l'action effectué
        current_actions_probs = tf.gather(tf.reshape(p, [-1]), indices)

        ratio = current_actions_probs / actions_probs
        clipped = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * deltas
        surrogate = tf.math.minimum(ratio * deltas, clipped)
        entropy = current_actions_probs * tf.math.log(current_actions_probs)
        policy_loss = - (tf.reduce_mean(surrogate) + c2 * tf.reduce_mean(entropy))

    with tf.GradientTape() as tape_value:
        v = value_model(states, training=True)
        value_loss = tf.keras.losses.mean_squared_error(v, discount_rewards)

    policy_grads = tape_policy.gradient(policy_loss, policy_model.trainable_variables)
    value_grads = tape_value.gradient(value_loss, value_model.trainable_variables)
    opt_policy.apply_gradients(zip(policy_grads, policy_model.trainable_variables))
    opt_value.apply_gradients(zip(value_grads, value_model.trainable_variables))

def train_model(policy_model, value_model, opt_policy, opt_value, buffer, epsilon, c2, epochs, n_steps, batch_size):
    buffer = np.array(buffer)
    # states = np.array(states)
    # masks = np.array(masks)
    # actions = np.array(actions)
    # discount_rewards = np.array(discount_rewards, dtype='float32')
    # values = np.array(values, dtype='float32')
    # deltas = discount_rewards - values
    # actions_probs = np.array(actions_probs, dtype='float32')

    for _ in range(epochs):
        buffer = np.random.permutation(buffer)
        states = np.vstack(buffer[:, 0])
        masks = np.vstack(buffer[:, 1])
        actions = np.array(buffer[:, 2], dtype='int32')
        discount_rewards = np.array(buffer[:, 4], dtype='float32')
        values = np.array(buffer[:, 3], dtype='float32')
        deltas = discount_rewards - values
        actions_probs = np.array(buffer[:, 5], dtype='float32')

        for i in range(0, n_steps, batch_size):
            my_train(policy_model, value_model, opt_policy, opt_value, states[i: i+batch_size], masks[i: i+batch_size],
                     actions[i: i+batch_size], deltas[i: i+batch_size], discount_rewards[i: i+batch_size],
                     actions_probs[i: i+batch_size], epsilon, c2)

def ppo_a2c(env: SingleAgentEnv,
                           gamma: float = 0.99999,
                           lr_policy: float = 0.001,
                           lr_value: float = 0.001,
                           epsilon: float = 0.2,
                           c2: float = 0.01,
                           epochs: int = 3,
                           batch_size: int = 32,
                           n_steps: int = 512,
                           max_episodes_count: int = 10000):

    assert(n_steps % batch_size == 0)
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init model
    policy_model = PolicyModel(env.state_size, env.action_size)
    value_model = ValueModel(env.state_size)

    optimizer_policy = Adam(learning_rate=lr_policy)
    optimizer_value = Adam(learning_rate=lr_value)

    buffer = deque(maxlen=n_steps)
    count_step = 0

    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0

        states = []
        masks = []
        rewards = []
        actions = []
        values = []
        actions_probs = []

        env.reset()
        while not env.is_game_over():
            s = env.state_vector()
            mask = env.available_actions_mask()

            v = value_model.predict(s.reshape(1, len(s)))
            pi = policy_model.predict(s.reshape(1, len(s)), mask.reshape(1, len(mask)))
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
            values.append(v.numpy()[0, 0])
            actions_probs.append(pi[a])

            lenght_episode += 1
            count_step += 1

        G = 0
        discount_rewards = []
        for t in reversed(range(len(states))):
            r_t = rewards[t]
            G = r_t + gamma * G
            discount_rewards.append(G)
        discount_rewards.reverse()
        discount_rewards = np.array(discount_rewards)

        for i in range(len(states)):
            buffer.append(np.array((states[i], masks[i], actions[i], values[i], discount_rewards[i], actions_probs[i]), dtype=object))

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

        if count_step >= n_steps:
            train_model(policy_model, value_model, optimizer_policy, optimizer_value, buffer, epsilon, c2, epochs, n_steps, batch_size)
            count_step = 0

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    logs_path = os.path.join('logs', env.__class__.__name__, 'ppo_a2c')
    logs_name = f'c2{c2}_epochs{epochs}_batchsize{batch_size}_nsteps{n_steps}_1m_logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/PPO_A2C/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    policy_model.save(model_save_path)