from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.DRL_algorithm.MCTS import Node, Policy_Player_MCTS

from src.agent_env.TicTacToeEnv import TicTacToeEnv

episodes = 100
rewards = []
moving_average = []

length_episodes = []
length_episodes_average = []

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



    len_episode = 0

    while not done:

        len_episode += 1

        mytree, action = Policy_Player_MCTS(mytree)

        print('action: ' + str(action))
        print('main available actions: ' + str(game.available_actions_ids()))

        print('mytree.N: ' + str(mytree.N))

        game.act_with_action_id(action)

        print('next available actions: ' + str(game.available_actions_ids()))

        observation = game.state_vector()

        print('mytree observation: ' + str(mytree.observation))

        print('observation: ' + str(observation))

        reward = game.score()

        done = game.is_game_over()

        reward_e = reward_e + reward

        new_game = deepcopy(game)

        if not done:

            mytree = Node(new_game, False, 0, observation, 0)

        # game.render() # uncomment this if you want to see your agent in action!

        if done:
            print('reward_e ' + str(reward_e))
            game.reset()
            break

    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-50:]))
    length_episodes.append(len_episode)
    length_episodes_average.append(np.mean(length_episodes[-50:]))

mean_rewards = np.mean(rewards)

plt.title('Rewards')
plt.plot(rewards, label='rewards')
plt.plot(moving_average, label='moving average')
plt.axline((0, mean_rewards), (len(rewards), mean_rewards), color='r', label='mean reward')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()

plt.title('length of episodes')
plt.plot(length_episodes, label='length of episodes')
plt.plot(length_episodes_average, label='moving average')
plt.xlabel('episodes')
plt.ylabel('length of episodes')
plt.show()
print('moving average: ' + str(np.mean(rewards[-20:])))
