from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from envs.tictactoe import TicTacToeLogic
import tensorflow as tf
import numpy as np


def acceptable_softmax_with_mask(X: np.ndarray, M: np.ndarray):
    positive_X = X - np.min(X)
    masked_positive_X = positive_X * M
    max_X = np.max(masked_positive_X)
    exp_X = np.exp(masked_positive_X - max_X)
    masked_exp_X = exp_X * M
    return masked_exp_X/np.sum(masked_exp_X)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.batch_size = 32

        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        self.model = Sequential([
            Dense(units=24, input_shape=(state_size,), activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))

    def compute_action(self, current_state, env):
        if np.random.uniform(0, 1) < self.exploration_proba:
            empty_cells = []
            for i, cell in enumerate(env.flatten_and_convert()):
                if cell == 0:
                    empty_cells.append(i)
            return np.random.choice(empty_cells)
        q_values = self.model.predict(current_state)[0]
        q_values_masked = acceptable_softmax_with_mask(q_values, env.get_mask())
        return np.argmax(q_values_masked)

    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        for experience in batch_sample:
            q_current_state = self.model.predict(experience["current_state"])
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma * np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            self.model.fit(x=experience["current_state"], y=q_current_state, verbose=0) # x=experience["current_state"][0], y=q_current_state[0]


if __name__ == '__main__':
    env = TicTacToeLogic()
    state_size = 9
    action_size = 9
    n_episodes = 50
    max_iteration_ep = 50
    agent = DQNAgent(state_size, action_size)

    for e in range(n_episodes):
        env.reset()
        current_state = np.array([env.flatten_and_convert()])
        for step in range(max_iteration_ep):
            action = agent.compute_action(current_state, env)
            next_state, reward, done = env.step(action)
            next_state = np.array([next_state])
            agent.store_episode(current_state, action, reward, next_state, done)
            current_state = next_state
            agent.train()
            # Update de decay exploration probability
            if done:
                agent.update_exploration_probability()
                break
    agent.model.save("./test.keras")
