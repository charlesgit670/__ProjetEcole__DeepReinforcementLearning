import logging
import random
import sys
import colorama


import environment

class Agent():
    
    def __init__(self, player_num):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('__init__')

        self.player_num = player_num

    def pick_action(self, environment):
        self.logger.debug('pick_action()')

        return None

class RandomAgent(Agent):

    def __init__(self, player_num):
        super().__init__(player_num)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('__init__')
        
    def pick_action(self, environment):
        self.logger.debug('pick_action()')

        if environment.actions:
            return random.choice(environment.actions)
        else:
            return None

class HumanAgent(Agent):

    def __init__(self, player_num):
        super().__init__(player_num)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('__init__')

    def pick_action(self, environment):
        self.logger.debug('pick_action()')

        # Print the current board.
        colorize = True    
        environment.render(colorize)

        selected_action = None

        # Repeat until a valid selection is made.
        while not selected_action:
            
            # Print the list of available actions as a menu.
            
            # Menu title:
            print()
            print((environment.PLAYER_INFO[environment.current_player][1] if colorize else "") + "Select action for player #{n} ({name}):".format(
                n = environment.current_player,
                name = environment.PLAYER_INFO[environment.current_player][0]) + (colorama.Fore.WHITE if colorize else "") 
            )

            # List of actions:
            i = 0
            for action in environment.actions:
                i += 1
                print('    {i}: {action}'.format(i = i, action = action[0]))
            
            # Add the fixed action to quit the program. That is extraneous to the environment.
            print('    x: Quit game')

            # Read input.
            proposed_action_index = input()
            try: 
                # Is it an integer?
                proposed_action_index_int = int(proposed_action_index)-1
                # Yes. Is the integer valid? 
                if (proposed_action_index_int >= 0) and (proposed_action_index_int < len(environment.actions)):
                    selected_action = environment.actions[proposed_action_index_int]

            except ValueError:
                # Not an integer. Check for x to quit. If not 'x' this is not a valid action. 
                if proposed_action_index.lower() == 'x':
                    selected_action = 'x'

            # Report mis-selection
            if not selected_action:
                print((colorama.Fore.RED if colorize else "") + "Invalid action selected, please try again.")

        # Act on the user-selection. Either quit the program or return the selected action to the runner. 
        if selected_action == 'x':
            self.logger.info('User selected quit')
            sys.exit(1)
        else:
            return selected_action

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            actions = self.model(state)
            return torch.argmax(actions).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.from_numpy(next_state).float().unsqueeze(0)
                target = (reward + self.gamma *
                          torch.max(self.model(next_state).detach()).item())
            state = torch.from_numpy(state).float().unsqueeze(0)
            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
