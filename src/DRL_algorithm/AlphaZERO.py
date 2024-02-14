import os
import numpy as np
import copy
import time
import json
from tqdm import tqdm
import joblib

from src.agent_env import SingleAgentEnv

import tensorflow as tf
from tensorflow import keras

from collections import namedtuple


tf.keras.backend.set_floatx('float64')

HIDDEN_STATES = 64

'''
We will use 2 Neural Networks for the algorithm implementation.
Note that can be also implemented as one single network sharing the same weights that will produce two outputs.
Also, often there is the usage of a CNN (Convolutional Neural Network) architecture in order to deal with the dynamic pixels of the game directly.
'''


class PolicyV(keras.Model):
    '''
    The Value Neural Network will approximate the Value of the node, given a State of the game.
    '''

    def __init__(self):
        super(PolicyV, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_1')

        self.dense2 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_2')

        self.v_out = keras.layers.Dense(1,
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='v_out')

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.v_out(x)

        return x


class PolicyP(keras.Model):
    '''
    The Policy Neural Network will approximate the MCTS policy for the choice of nodes, given a State of the game.
    '''

    def __init__(self):
        super(PolicyP, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_1')

        self.dense2 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_2')

        self.p_out = keras.layers.Dense(GAME_ACTIONS,
                                        activation='softmax',
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='p_out')

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.p_out(x)

        return x

class Node:
    def __init__(self,
                 action: int = None,
                 parent: 'Node' = None):
        self.parent = parent
        self.action = action
        self.child = []
        self.T = 0 # total rewards from MCTS exploration
        self.N = 0 # visit count
        self.avails = 1 # number of time this node was a legal move

    def getUCB_best_child(self, legal_actions, c):

        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.
        '''

        # keep only children in legal actions
        legal_children = [child for child in self.child if child.action in legal_actions]
        # log_total_visits = np.log(np.sum(child.N for child in legal_children))

        # We use one of the possible MCTS formula for calculating the node value
        def ucb_score(child):
            return (child.T / child.N) + c * np.sqrt(np.log(child.avails) / child.N)

        for child in legal_children:
            child.avails += 1

        return max(legal_children, key=ucb_score)

    def get_untried_actions(self, legal_actions):
        """ Return the elements of legal_actions for which this node does not have children.
        """

        # Find all moves for which this node *does* have children
        tried_actions = [child.action for child in self.child]

        # Return all moves that are legal but have not been tried yet
        return [a for a in legal_actions if a not in tried_actions]

    def create_child(self, action):
        """ Add a new child node for the action.
            Return the added child node
        """
        child = Node(action, self)
        self.child.append(child)
        return child

    def rollout(self, env):
        '''
        The rollout is a random play from a copy of the environment of the current node using random moves.
        This will give us a value for the current node.
        Taken alone, this value is quite random, but, the more rollouts we will do for such node,
        the more accurate the average of the value for such node will be. This is at the core of the MCTS algorithm.
        '''
        while not env.is_game_over():
            aa = env.available_actions_ids()
            action = np.random.choice(aa)
            env.act_with_action_id(action)

        return env.score()

    def explore(self, env, c, max_time=1):

        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        Repeat until time is reached
        '''

        start_time = time.time()
        while time.time() - start_time < max_time:
            new_env = copy.deepcopy(env)
            # SELECTION
            current = self
            while not new_env.is_game_over() and current.get_untried_actions(new_env.available_actions_ids()) == []:
                current = current.getUCB_best_child(new_env.available_actions_ids(), c)
                new_env.act_with_action_id(current.action)

            # EXPANSION
            untried_actions = current.get_untried_actions(new_env.available_actions_ids())
            if untried_actions != []:
                a = np.random.choice(untried_actions)
                new_env.act_with_action_id(a)
                current = current.create_child(a)

            # SIMULATION
            r = current.rollout(new_env)
            current.T += r
            current.N += 1

            # BACKPROPAGATION
            parent = current

            while parent.parent:
                parent = parent.parent
                parent.N += 1
                parent.T += r

    def next(self, legal_actions):
        '''
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        # verify child  are in  available_actions_ids
        legal_child = [child for child in self.child if child.action in legal_actions]

        if not legal_child:
            raise ValueError('no children found and game hasn\'t ended')

        child = max(legal_child, key=lambda c: c.N)

        return child, child.action



'''
The ReplayBuffer stores game plays that we will use for neural network training. It stores, in particular:
    - The observation (i.e. state) of the game environment
    - The target Value
    - The observation (i.e. state) of the game environment at the previous step
    - The target Policy according to visit counts 
'''


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["obs", "v", "p_obs", "p"])

    def add(self, obs, v, p, p_obs):
        """Add a new experience to memory."""

        e = self.experience(obs, v, p, p_obs)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        return experiences

    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)



def ISMCTS(env: SingleAgentEnv,
           gamma: float = 0.99999,
           c: float = np.sqrt(2),
           time_per_action: float = 0.1,
           max_episodes_count: int = 100):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init MCTS
    env.reset()
    # ref_root = Node()
    # MCTS_tree = ref_root
    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0
        env.reset()
        # MCTS_tree = ref_root
        while not env.is_game_over():
            MCTS_tree = Node()
            MCTS_tree.explore(copy.deepcopy(env), c, time_per_action)
            MCTS_tree, a = MCTS_tree.next(env.available_actions_ids())

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

    # # save logs
    # dict_logs = {
    #     "lenght_episodes": lenght_episodes,
    #     "reward_episodes": reward_episodes
    # }
    # logs_path = os.path.join('logs', env.__class__.__name__, 'ISMCTS')
    # logs_name = 'logs.json'
    # if not os.path.exists(logs_path):
    #     os.makedirs(logs_path)
    # with open(os.path.join(logs_path, logs_name), 'w') as file:
    #     json.dump(dict_logs, file)
    #
    # model_save_path = 'model/ISMCTS/'
    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)
    # joblib.dump(ref_root, os.path.join(model_save_path, 'ISMCTS_Tree_Object.joblib'))