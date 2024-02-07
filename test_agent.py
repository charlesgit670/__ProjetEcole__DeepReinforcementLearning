from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.DRL_algorithm.MCTS import Node, Policy_Player_MCTS

from src.agent_env.TicTacToeEnv import TicTacToeEnv

episodes = 10
rewards = []
moving_average = []

'''
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our MCTS algorithm
- we will collect and plot the rewards to check if the MCTS is actually working.
- For CartPole-v0, in particular, 200 is the maximum possible reward. 
'''

for e in range(episodes):

    reward_e = 0
    game = TicTacToeEnv()
    observation = game.reset()
    done = False

    new_game = deepcopy(game)
    mytree = Node(new_game, False, 0, observation, 0)

    print('episode #' + str(e + 1))

    while not done:

        mytree, action = Policy_Player_MCTS(mytree)

        print('action: ' + str(action))
        print('main available actions: ' + str(game.available_actions_ids()))

        print('mytree.N: ' + str(mytree.N))

        game.act_with_action_id(action)

        print('next available actions: ' + str(game.available_actions_ids()))

        observation = game.state_vector()

        print('mytree observation: ' + str(mytree.observation))

        print('observation: ' + str(observation))

        print('parent : ', mytree.parent)

        reward = game.score()

        done = game.is_game_over()

        reward_e = reward_e + reward

        new_game = deepcopy(game)

        mytree = Node(new_game, False, 0, observation, 0)

        # game.render() # uncomment this if you want to see your agent in action!

        if done:
            print('reward_e ' + str(reward_e))
            game.reset()
            break

    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

plt.plot(rewards)
plt.plot(moving_average)
plt.show()
print('moving average: ' + str(np.mean(rewards[-20:])))
