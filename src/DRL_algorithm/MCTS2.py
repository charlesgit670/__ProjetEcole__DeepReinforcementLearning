import os
import numpy as np
import copy
import time
import json
from tqdm import tqdm
import joblib

from src.agent_env import SingleAgentEnv

class Node:
    def __init__(self,
                 # env: SingleAgentEnv,
                 states: np.array,
                 done: bool,
                 parent: 'Node' = None):
        self.parent = parent
        self.child = None
        self.T = 0 # total rewards from MCTS exploration
        self.N = 0 # visit count
        self.states = states
        self.done = done

    def getUCBscore(self, c):

        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.
        '''

        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')

        # We use one of the possible MCTS formula for calculating the node value
        return (self.T / self.N) + c * np.sqrt(np.log(self.parent.N) / self.N)

    def create_child(self, env):

        '''
        We create one children for each possible action of the game,
        then we apply such action to a copy of the current node enviroment
        and create such child node with proper information returned from the action executed
        '''

        if self.done:
            return

        new_env = copy.copy(env)
        new_env.reset_with_states(self.states)
        actions = new_env.available_actions_ids()

        child = {}
        for action in actions:
            new_env.reset_with_states(self.states)
            new_env.act_with_action_id(action)
            states = new_env.state_vector()
            done = new_env.is_game_over()

            child[action] = Node(states, done, self)

        self.child = child

    def rollout(self, env):

        '''
        The rollout is a random play from a copy of the environment of the current node using random moves.
        This will give us a value for the current node.
        Taken alone, this value is quite random, but, the more rollouts we will do for such node,
        the more accurate the average of the value for such node will be. This is at the core of the MCTS algorithm.
        '''

        if self.done:
            return 0

        reward = 0
        done = False
        # init env with current states
        new_env = copy.copy(env)
        new_env.reset_with_states(self.states)
        while not done:
            aa = new_env.available_actions_ids()
            action = np.random.choice(aa)

            old_score = new_env.score()
            new_env.act_with_action_id(action)
            new_score = new_env.score()
            reward = new_score - old_score

            done = new_env.is_game_over()

        return reward

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
            # SELECTION
            # find a leaf node by choosing nodes with max U.
            current = self

            while current.child:
                childs = current.child
                max_U = max(child.getUCBscore(c) for child in childs.values())
                actions = [a for a, child in childs.items() if child.getUCBscore(c) == max_U]

                action = np.random.choice(actions)
                current = childs[action]

            # play a random game, or expand if needed
            if current.N < 1:
                # SIMULATION
                current.T = current.T + current.rollout(env)
            else:
                # EXPANSION
                current.create_child(env)
                if current.child:
                    index = (list(current.child))
                    rand_index = np.random.choice(index)
                    current = current.child[rand_index]

                # SIMULATION
                current.T = current.T + current.rollout(env)

            current.N += 1

            # BACKPROPAGATION
            parent = current

            while parent.parent:
                parent = parent.parent
                parent.N += 1
                parent.T = parent.T + current.T

    def next(self):

        '''
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')

        child = self.child

        # verify child  are in  available_actions_ids
        # child = {a: c for a, c in child.items() if a in env.available_actions_ids()}

        max_N = max(node.N for node in child.values())
        max_children_actions = [a for a, c in child.items() if c.N == max_N]
        action = np.random.choice(max_children_actions)

        return child[action], action

def MCTS(env: SingleAgentEnv,
         gamma: float = 0.99999,
         c: float = np.sqrt(2),
         time_per_action: float = 0.1,

         max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    # init MCTS
    env.reset()
    MCTS_tree = Node(env.state_vector(), env.is_game_over(), None)
    ref_root = MCTS_tree
    for ep_id in tqdm(range(max_episodes_count)):
        lenght_episode = 0
        G = 0
        env.reset()
        MCTS_tree = ref_root
        while not env.is_game_over():
            MCTS_tree.explore(copy.copy(env), c, time_per_action)
            MCTS_tree, a = MCTS_tree.next()

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            G += gamma ** lenght_episode * r
            lenght_episode += 1

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    logs_path = os.path.join('logs', env.__class__.__name__, 'MCTS')
    logs_name = 'logs.json'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(os.path.join(logs_path, logs_name), 'w') as file:
        json.dump(dict_logs, file)

    model_save_path = 'model/MCTS/MCTS_Tree_Object.joblib'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    joblib.dump(ref_root, model_save_path)