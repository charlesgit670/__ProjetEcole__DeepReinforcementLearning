import os
import numpy as np
import json
from tqdm import tqdm

from src.agent_env import SingleAgentEnv
from src.DRL_algorithm.function_utils import argmax


def q_learning(env: SingleAgentEnv,
               gamma: float = 0.99999,
               alpha: float = 0.1,
               epsilon: float = 0.2,
               max_episodes_count: int = 10000):
    assert (epsilon > 0)
    assert (alpha > 0)

    # used for logs
    lenght_episodes = []
    reward_episodes = []

    pi = {}
    Q = {}

    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            # initialize pi[s], Q[s] if s is new
            if s not in pi.keys():
                pi[s] = {a: 0 for a in aa}
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                a = argmax(Q[s])

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()
            if s_p not in pi.keys():
                pi[s_p] = {a: 0 for a in aa_p}
                Q[s_p] = {a: np.random.uniform(-1.0, 1.0) for a in aa_p}

            if env.is_game_over():
                Q[s_p] = {}
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                Q_max = max(Q[s_p].values())
                Q[s][a] += alpha * (r + gamma * Q_max - Q[s][a])

            pi[s] = dict.fromkeys(pi[s], 0)
            pi[s][argmax(Q[s])] = 1.0

            G += gamma ** lenght_episode * r
            lenght_episode += 1
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    print(len(pi))
    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    logs_path = os.path.join('logs', env.__class__.__name__, 'q_learning')
    logs_name = 'logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    ans = dict(sorted(pi.items())), dict(sorted(Q.items()))
    return ans