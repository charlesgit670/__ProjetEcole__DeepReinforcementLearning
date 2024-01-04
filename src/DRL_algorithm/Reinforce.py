import os
import json
import numpy as np
from tqdm import tqdm

from src.agent_env import SingleAgentEnv
# https://medium.com/nerd-for-tech/reinforcement-learning-introduction-to-policy-gradients-aa2ff134c1b
def reinforce(env: SingleAgentEnv,
              gamma: float = 0.99999,
              max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []


    Returns = {}
    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0

        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            #initialize pi[s], Q[s] and Returns[s] if s is new
            if s not in pi.keys():
                pi[s] = {a: 1/len(aa) for a in aa}
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}
                Returns[s] = {a: np.zeros(2) for a in aa}

            pi_s = [pi[s][a] for a in aa]
            assert(abs(np.sum(pi_s) - 1) < 1e-9)
            a = np.random.choice(aa, p=pi_s)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            S.append(s)
            A.append(a)
            R.append(r)
            lenght_episode += 1

        G = 0
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            r_t = R[t]

            G = r_t + gamma * G
            if (s_t, a_t) not in zip(S[0: t], A[0: t]):
                Returns[s_t][a_t][0] = (Returns[s_t][a_t][1]*Returns[s_t][a_t][0] + G)/(Returns[s_t][a_t][1] + 1)
                Returns[s_t][a_t][1] += 1
                Q[s_t][a_t] = Returns[s_t][a_t][0]
                best_a = argmax(Q[s_t])
                pi[s_t] = dict.fromkeys(pi[s_t], epsilon / len(pi[s_t]))
                pi[s_t][best_a] += 1 - epsilon
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    print(len(pi))
    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'on_policy_first_visit_monte_carlo_control_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = dict(sorted(pi.items())), dict(sorted(Q.items()))
    return ans