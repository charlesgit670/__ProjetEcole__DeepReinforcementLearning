import os
import json
import numpy as np
from tqdm import tqdm
import time


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.DRL_algorithm.function_utils import apply_mask
from src.agent_env import SingleAgentEnv

from src.agent_env.BalloonPop.main import BalloonPOPEnv
from src.agent_env.CantStopML.src.main import CantStopGame

def play_random(env :SingleAgentEnv):
    '''
    This function is used to play the game with a random agent.
    It is used to test the environment and the game rules.
    '''
    env.reset()
    while not env.is_game_over():
        s = env.state_vector()

        aa = env.available_actions_ids()
        mask = env.available_actions_mask()

        flat_aa = np.array(aa).flatten()

        # selected_index = np.random.choice(aa.shape[0])
        # a  = aa[selected_index]

        print(aa)

        a = np.random.choice(aa)
        # view_action = env.number_action_for4dice[a]
        env.act_with_action_id(a)


    return env.score()

def main():
    # env = BalloonPOPEnv()
    env = CantStopGame(logs=False)
    score = []
    start_time = time.time()
    for i  in range(1_000):
        score.append(play_random(env))

        print(f'game {i} score: {score[-1]}')

    print(f'elapsed time: {time.time() - start_time}')

    print('moy score:', np.mean(score))

def main_timed():
    start_time = time.time()
    games_played = 0
    env = BalloonPOPEnv()
    score = []

    while time.time() - start_time < 1:  # Loop until one second has passed
        score.append(play_random(env))  # Play a single game
        games_played += 1

    print(f"Number of games played in 1 second: {games_played}")
    print('moy score:', np.mean(score))
    #2600 - 2700 games played in 1 second with random agent on BalloonPOPEnv
    #average random player score = 42

if __name__ == '__main__':
    main()
    # main_timed()