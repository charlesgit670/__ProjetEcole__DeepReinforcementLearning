import os
import json
import numpy as np
from tqdm import tqdm
import time
from math import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.DRL_algorithm.function_utils import timing_decorator, apply_mask
from src.agent_env import SingleAgentEnv

import numpy as np
import random
from collections import defaultdict, deque
from copy import deepcopy

c = sqrt(2)    # MCTS exploring constant: the higher, the more reliable, but slower in execution time


class Node:
    '''
    The Node class represents a node of the MCTS tree.
    It contains the information needed for the algorithm to run its search.
    '''

    def __init__(self, game, done, parent, observation, action_index):

        # child nodes
        self.child = None

        # total rewards from MCTS exploration
        self.T = 0

        # visit count
        self.N = 0

        # the environment
        self.game = game

        # observation of the environment
        self.observation = observation

        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent

        # action index that leads to this node
        self.action_index = action_index

    def getUCBscore(self):

        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.
        '''

        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')

        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        # We use one of the possible MCTS formula for calculating the node value
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N)

    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None

    def create_child(self):

        '''
        We create one children for each possible action of the game,
        then we apply such action to a copy of the current node enviroment
        and create such child node with proper information returned from the action executed
        '''

        if self.done:
            return

        actions = []
        games = []

        for i in self.game.available_actions_ids():

            actions.append(i)
            new_game = deepcopy(self.game)
            games.append(new_game)

        child = {}
        for action, game in zip(actions, games):


            game.act_with_action_id(action)

            observation = game.state_vector()

            reward = game.score()

            done = game.is_game_over()

            child[action] = Node(game, done, self, observation, action)

        self.child = child

    def explore(self):

        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        '''

        # find a leaf node by choosing nodes with max U.

        current = self

        while current.child:



            child = current.child

            for c in child.values():
                print('child.N: ' + str(c.N))
                print('child.T: ' + str(c.T))
                print('child.action_index: ' + str(c.action_index))
                print('ucb score: ' + str(c.getUCBscore()))


            max_U = max(c.getUCBscore() for c in child.values())

            print('max_U: ' + str(max_U))



            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]





            if len(actions) == 0:
                print("error zero length ", max_U)

            action = random.choice(actions)


            current = child[action]

        # play a random game, or expand if needed

        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:


                index = (list(current.child))


                rand_index = random.choice(index)


                current = current.child[rand_index]

            current.T = current.T + current.rollout()

        current.N += 1

        # update statistics and backpropagate

        parent = current

        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):

        '''
        The rollout is a random play from a copy of the environment of the current node using random moves.
        This will give us a value for the current node.
        Taken alone, this value is quite random, but, the more rollouts we will do for such node,
        the more accurate the average of the value for such node will be. This is at the core of the MCTS algorithm.
        '''

        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.game)
        while not done:
            aa = new_game.available_actions_ids()


            action = np.random.choice(aa)

            new_game.act_with_action_id(action)

            observation = new_game.state_vector()

            done = new_game.is_game_over()

            if done:
                reward = new_game.score()

                v = v + reward

                new_game.reset()

                break
        return v

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



        child = {a: c for a, c in child.items() if a in self.game.available_actions_ids()}


        max_N = max(node.N for node in child.values())


        #verify child = are in  available_actions_ids


        max_children = [c for a, c in child.items() if c.N == max_N]

        if len(max_children) == 0:
            print("error zero length ", max_N)




        max_child = random.choice(max_children)


        return max_child, max_child.action_index


MCTS_POLICY_EXPLORE = 200_000  # MCTS exploring constant: the higher, the more reliable, but slower in execution time


def Policy_Player_MCTS(mytree):
    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
#add timer
    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()




    next_tree, next_action = mytree.next()

    print('mytree.N: ' + str(next_tree.N))
    print('mytree.T: ' + str(next_tree.T))
    print('observation: ' + str(next_tree.observation))
    print('action: ' + str(next_action))


    # note that here we are detaching the current node and returning the sub-tree
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()

    return next_tree, next_action