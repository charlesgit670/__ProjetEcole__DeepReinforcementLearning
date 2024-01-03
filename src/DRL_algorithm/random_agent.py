import numpy as np
import json
from tqdm import tqdm

from src.agent_env import SingleAgentEnv


def random_evaluation(env: SingleAgentEnv,
                      gamma: float = 0.99999,
                      is_reset_random=False,
                      max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []
    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0

        if is_reset_random:
            env.reset_random()
        else:
            env.reset()
        if env.is_game_over():
            continue
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()
            a = np.random.choice(aa)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            G += gamma ** lenght_episode * r
            lenght_episode += 1

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    #save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open('logs/random_evaluation.json', 'w') as file:
        json.dump(dict_logs, file)

    return None