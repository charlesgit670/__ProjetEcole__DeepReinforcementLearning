import random
import time
import numpy as np
import copy

from src.agent_env.SingleAgentEnv import SingleAgentDeepEnv



class CantStopGame(SingleAgentDeepEnv):

    def __init__(self, logs: bool = True):
        self.board = self.initialize_board()
        self.current_player = 0
        self.blocked_columns = [0, 0]
        # self.active_columns = set()
        self.last_dice_roll = []
        self.is_over = False
        self.logs = logs

        ###TBD
        self.state_size = 18
        self.action_size = 9
        self.states = np.zeros(9)
        self.play_first = play_first
        self.reset()
        ###TBD

        def state_vector(self) -> np.array: