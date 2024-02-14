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

    def explore(self, env, c, max_iteration):

        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        Repeat until time is reached
        '''

        # start_time = time.time()
        # while time.time() - start_time < max_time:
        for _ in range(max_iteration):
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

def ISMCTS(env: SingleAgentEnv,
           gamma: float = 0.99999,
           c: float = np.sqrt(2),
           max_iteration: int = 1000,
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
            MCTS_tree.explore(copy.deepcopy(env), c, max_iteration)
            MCTS_tree, a = MCTS_tree.next(env.available_actions_ids())

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            G += gamma ** lenght_episode * r
            lenght_episode += 1

        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    print(f"With {max_iteration} iterations during exploration we got :")
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