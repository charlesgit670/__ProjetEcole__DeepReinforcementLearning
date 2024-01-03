# This is a scaffolding using an instance of Environment allowing it to be played
# by humans in a console window. 

import logging
import argparse
import random
import sys
import os
import colorama
import time

import environment
import agent as ag

options = None

# Conversion function for argparse booleans
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Set up argparse and get the command line options.
def parse_commandline():

    global options

    parser = argparse.ArgumentParser(
        description = 'This is an implementation of the board game "can''t stop".', 
        epilog = 'Remember to be nice!'
    )

    parser.add_argument('-ll', '--log-level',
        action = 'store',
        default = 'INFO',
        help ='Set the logging output level to CRITICAL, ERROR, WARNING, INFO or DEBUG (default: %(default)s)',
        dest ='log_level',
        metavar = 'level'
    )

    parser.add_argument('-co', '--colorize',
        action = 'store',
        default = True,
        type = str2bool,
        help ='Select colorized output (default: %(default)s)',
        dest ='colorize',
        metavar = 'colorize'
    )

    parser.add_argument('-rs', '--random-seed',
        action = 'store',
        default = None,
        help = 'Set random seed to make dice rolls reproducible',
        dest ='randseed',
        metavar = 'randseed'
    )

    options = parser.parse_args()
    options.log_level_int = getattr(logging, options.log_level, logging.INFO)

    if options.randseed:
        random.seed(options.randseed)

# Set up a logger each for a file in the output folder and the console.      
def setup_logging():
  
    global options
  
    fh = logging.FileHandler(os.path.dirname(os.path.realpath(__file__)) + '\\manual.log')
    fh.setLevel(options.log_level_int)

    ch = logging.StreamHandler()
    ch.setLevel(options.log_level_int)

    ch.setFormatter(logging.Formatter('({thread}) [{levelname:7}] {name} - {message}', style='{'))
    fh.setFormatter(logging.Formatter('{asctime} ({thread}) [{levelname:7}] {name} - {message}', style='{'))

    root = logging.getLogger()
    root.addHandler(ch)
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)

def main(headless=False):
    global options

    # So we can redirect full Unicode even in Windows.
    sys.stdout.reconfigure(encoding='utf-8')

    parse_commandline()
    setup_logging()
    #
    logger = logging.getLogger('Main')
    logger.info('Starting.')

    # Create and reset the environment
    env = environment.Environment()
    env.reset()




    # Set up agents:
    agents = [
        ag.RandomAgent(0),
        ag.HumanAgent(2)
    ]

    #human_index = random.randint(0,2)
    #agents[human_index] = ag.HumanAgent(human_index)

    while (env.winner == -1):
        
        # Print the current board.    
        # env.render(options.colorize)

        # Find the agent whose turn it is.
        agent = agents[env.current_player]



        # Ask the agent for an action
        action = agent.pick_action(env)

        if action:
            if not headless:

                print((env.PLAYER_INFO[env.current_player][1] if options.colorize else "") + "Selected action: Player #{current_player} ({name}) plays: {action}.".format(
                    current_player = env.current_player,
                    name = env.PLAYER_INFO[env.current_player][0],
                    action = action[0]) + (colorama.Fore.WHITE if options.colorize else ""))



            # board = env.board
            # print(board)


            # Take the action
            env.take_action(action)
        else:
            raise ValueError("Agent did not pick an action")

    if not headless:
        # Render the final board.
        env.render(options.colorize)

    # logger.info('Finished.')

from agent import DQNAgent

def benchmark():
    """Benchmark the game to determine how many games can be played in one second."""

    global options

    # So we can redirect full Unicode even in Windows.
    sys.stdout.reconfigure(encoding='utf-8')

    parse_commandline()
    setup_logging()
    #
    logger = logging.getLogger('Main')
    logger.info('Starting.')

    # Create and reset the environment
    env = environment.Environment()

    start_time = time.time()
    count = 0

    # Set up agents:
    agents = [
        ag.RandomAgent(0),
        ag.HumanAgent(2)
    ]

    while time.time() - start_time < 1.0:

        env.reset()



        # human_index = random.randint(0,2)
        # agents[human_index] = ag.HumanAgent(human_index)

        while (env.winner == -1):

            # Print the current board.
            # env.render(options.colorize)

            # Find the agent whose turn it is.
            agent = agents[env.current_player]

            # Ask the agent for an action
            action = agent.pick_action(env)

            if action:


                # board = env.board
                # print(board)

                # Take the action
                env.take_action(action)
            else:
                raise ValueError("Agent did not pick an action")

        # if not headless:
        #     # Render the final board.
        env.render(options.colorize)

        count += 1
    print(f"Played {count} games in one second.")
    return count
    
if __name__ == '__main__':
    # main_train()
    # for i in range(1000):
    # count = benchmark()
    main()







def main_train():
    # Paramètres
    state_size = 4  # Remplacez par la taille de l'état de votre environnement
    action_size = 2  # Remplacez par le nombre d'actions possibles dans votre environnement
    num_episodes = 1000  # Nombre total d'épisodes pour l'entraînement
    batch_size = 32  # Taille du lot pour l'entraînement

    # Initialisation de l'environnement et de l'agent
    env = environment.Environment()  # Remplacez par votre classe Environment
    agents = [
        ag.RandomAgent(0),
        ag.DQNAgent(state_size, action_size)
    ]

    # Boucle principale d'entraînement
    for episode in range(num_episodes):
        state = env.reset()  # Réinitialise l'environnement pour un nouvel épisode
        total_reward = 0

        while (env.winner == -1):  # Remplacez par votre condition de fin de partie
            agent = agents[env.current_player]

            action = agent.act(state)  # Sélectionne une action

            next_state, reward, done, _ = env.step(action)  # Exécute l'action

            # Update only if the current agent is the DQN agent
            if isinstance(current_agent, DQNAgent):
                current_agent.remember(state, action, reward, next_state, done)
                if len(current_agent.memory) > batch_size:
                    current_agent.replay(batch_size)  # Train the neural network

            state = next_state  # Met à jour l'état

            total_reward += reward

            if done:  # Vérifie si la partie est finie
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # Entraîne le réseau de neurones

        # Ajoutez ici d'autres mécanismes de sauvegarde ou de suivi des performances

