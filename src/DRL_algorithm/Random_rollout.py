import copy
import time
import numpy as np
import json
from tqdm import tqdm

from src.agent_env import SingleAgentEnv

def random_rollout(env: SingleAgentEnv, max_time=1):
    new_env = copy.copy(env)
    init_states = env.state_vector()
    resultat_storage = {a: (0, 0) for a in env.available_actions_ids()}

    start_time = time.time()
    while time.time() - start_time < max_time:
        new_env.reset_with_states(init_states)
        aa = new_env.available_actions_ids()
        first_a = np.random.choice(aa)
        new_env.act_with_action_id(first_a)
        while not new_env.is_game_over():
            aa = new_env.available_actions_ids()
            a = np.random.choice(aa)
            new_env.act_with_action_id(a)
        tuple_element = resultat_storage[first_a]
        resultat_storage[first_a] = (tuple_element[0] + 1, tuple_element[1] + new_env.score())

    action_played = [k for k in resultat_storage if resultat_storage[k][0] != 0]
    best_action = max(action_played, key=lambda k: resultat_storage[k][1] / resultat_storage[k][0])
    return best_action

def random_rollout_evaluation(env: SingleAgentEnv, gamma: float = 0.99999, time_per_action: float = 0.1, max_episodes_count: int = 100):
    # used for logs
    lenght_episodes = []
    reward_episodes = []
    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0
        env.reset()
        while not env.is_game_over():
            a = random_rollout(env, max_time=time_per_action)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            G += gamma ** lenght_episode * r
            lenght_episode += 1

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    print(f"With a {time_per_action} seconds by action we got :")
    print(f"Mean score {round(np.mean(reward_episodes), 2)}")
    print(f"Mean episode lenght {round(np.mean(lenght_episodes), 2)}")