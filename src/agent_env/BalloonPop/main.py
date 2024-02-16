import random
from collections import Counter
import time
import numpy as np
import itertools

from src.agent_env.SingleAgentEnv import SingleAgentDeepEnv

class BalloonPOPEnv(SingleAgentDeepEnv):

    def __init__(self):
        # Constants
        self.BALLOON_COLORS = ["Yellow", "Blue", "Red"]
        self.BALLOON_SHAPES = ["Star", "Moon", "Kite"]

        self.NUM_DICE = 5
        self.NUM_MAX_BREAKS = 3

        self.BALLOONS = self.BALLOON_COLORS + self.BALLOON_SHAPES

        self.BUST_LIMITS = {
            "Yellow": 6,
            "Blue": 7,
            "Red": 10,
            "Star": 9,
            "Moon": 8,
            "Kite": 6
        }
        self.bust_limits_colors = np.array([6, 7, 10])
        self.bust_limits_shape = np.array([9, 8, 6])

        self.bust_limits_all = np.vstack((self.bust_limits_colors, self.bust_limits_shape))


        self.DICE_FACE = {
            1: [self.BALLOON_COLORS[2], self.BALLOON_SHAPES[0]],
            2: [self.BALLOON_COLORS[2], self.BALLOON_SHAPES[1]],
            3: [self.BALLOON_COLORS[2], self.BALLOON_SHAPES[2]],
            4: [self.BALLOON_COLORS[1], self.BALLOON_SHAPES[0]],
            5: [self.BALLOON_COLORS[1], self.BALLOON_SHAPES[1]],
            6: [self.BALLOON_COLORS[0], self.BALLOON_SHAPES[0]],
        }


        self.BALLOON_SCORES_COLORS = np.array([np.array([0,3,7,11,15,3]), np.array([1,3,5,7,9,12,8]), np.array([0,0,0,2,4,6,8,10,14,6])], dtype=object)
        self.BALLOON_SCORES_SHAPES = np.array([np.array([1,2,3,5,7,10,13,16,4]), np.array([2,3,4,5,7,9,12,5]), np.array([1,3,6,10,13,7])], dtype=object)
        self.BALLOON_SCORES = np.vstack((self.BALLOON_SCORES_COLORS, self.BALLOON_SCORES_SHAPES))

        self.num_breaks = 0

        self.state_size = 16 #6 ballons [2, 1, 3, 1, 0, 4] et 5 dés [color [0 || 1 || 2 || 3], shape [0 || 1 || 2 || 3]] * 5
        # one hot state size [[0, 0, 1], [0, 1, 0], [1, 0, 0]] * 6 + [[0, 0, 1], [0, 1, 0], [1, 0, 0]] * 5
        self.action_size = 16 # si le joueur décide de relancer les dés 1, 3 et 4, le vecteur serait : [1, 0, 1, 1, 0]
        self.states_balloons = np.zeros((2,3), dtype=int)
        self.states_dice = np.zeros((5,2), dtype=int)
        self.states_dice[:3, :2] = np.random.randint(1, 4, size=(3, 2))
        self.num_dice = 3
        self.total_score = 0

        self.list_total_score = np.array([])

        self.scoring = 0

        self.number_action_for3dice = self.generate_binary_numbers(3)
        self.number_action_for4dice = self.generate_binary_numbers(4)



        self.reset()

    def generate_binary_numbers(self, n):
        # Create an iterable of 0 and 1
        bits = [0, 1]
        # Use itertools.product to generate all combinations, specifying repeat=n
        binary_numbers = itertools.product(bits, repeat=n)
        # Convert each combination to a binary string and return the list
        initial_binary_numbers = (list(number) for number in binary_numbers)


        new_numpy = []

        for i in initial_binary_numbers:

            if i[-1]:
                new_numpy.append(i)

            else:
                new_numpy.insert(0, i)

        return new_numpy

    def state_vector(self) -> np.array:

        # onehotdice = self.dice_to_onehot(self.states_dice)
        #
        # onehotboard = self.board_to_onehot(self.states_balloons)

        # score_in_state_balloon = self.board_to_scored_board()

        ##TODO: 2. Implement one hot encoding for the state vector and state dice and state_size
        # state_vector = np.concatenate((self.states_dice, score_in_state_balloon), axis=None)

        state_vector = np.concatenate((self.states_dice, self.states_balloons), axis=None)
        # state_vector = np.concatenate((onehotdice, onehotboard), axis=None)

        return np.array(state_vector).flatten()


    def board_to_scored_board(self):
        state_score_value = np.array([])

        for index, value in enumerate(self.states_balloons):

            for index2, value2 in enumerate(value):


                state_score_value = np.append(state_score_value, self.BALLOON_SCORES[index][index2][value2 - 1])

        return state_score_value


    def board_to_onehot(self, states_balloons):

        onehottotal = np.array([], dtype=int)

        bust_limits_flatten = np.array(list(self.BUST_LIMITS.values())).flatten()

        for index, column in enumerate(np.array(states_balloons).flatten()):

            len_col = bust_limits_flatten[index]

            onehotcolumn = np.zeros((len_col + 1), dtype=int)

            onehotcolumn[column] = 1

            onehottotal = np.concatenate((onehottotal, onehotcolumn))

        return onehottotal

    def dice_to_onehot(self, state_dice):

        onehottotal = np.array([], dtype=int)

        for color, shape in state_dice:
            onehotstatecolor = np.zeros((4), dtype=int)

            onehotstatecolor[color] = 1

            onehotstateshape = np.zeros((4), dtype=int)

            onehotstateshape[shape] = 1

            onehotdice = np.concatenate((onehotstatecolor, onehotstateshape), dtype=int)

            onehottotal = np.concatenate((onehottotal, onehotdice))

        return onehottotal

    def is_game_over(self) -> bool:

        if self.num_breaks >= self.NUM_MAX_BREAKS:
            return True

        return False

    def act_with_action_id(self, action_id): #action_id: [0, 1, 0, 1, 0]
        if self.is_game_over():
            return


        if self.num_dice <= 4:
            vector_action_id = self.number_action_for4dice[action_id]

            action_mask_for_id = self.dice_mask()

            to_play = vector_action_id * action_mask_for_id  # [1, 0, 1, 1, 1] * [1, 1, 1, 0, 0] = [1, 0, 1, 0, 0]

        else:
            raise ValueError(f'number dice false {self.num_dice} ')


        #convert to_play to numpy array
        to_play = np.array(to_play)


        if np.any(to_play == 1) and self.num_dice < 5:

            self.num_dice += 1

            # add new dice value to the states_dice

            self.states_dice[self.num_dice - 1, :2] = np.random.randint(1, 4, size=(1, 2))

            if not np.any(self.number_action_for4dice == to_play):
                # Properly format the error message
                available_actions = self.available_actions_ids()
                error_message = f'DICE {to_play}  is not in the available actions: {available_actions}'
                raise ValueError(error_message)

            for index, value in enumerate(to_play):

                if value == 1:


                    self.states_dice[index, :2] = np.random.randint(1, 4, size=(1, 2))

            if self.num_dice == 5:


                self.play_dice_on_balloon()
                self.dice_reset()

        elif np.any(to_play == 0) and self.num_dice == 3:

            self.states_dice = np.random.randint(1, 4, size=self.states_dice.shape)

            self.play_dice_on_balloon()
            self.dice_reset()

        elif np.any(to_play == 0) and self.num_dice > 3:

            self.play_dice_on_balloon()
            self.dice_reset()

        elif self.num_dice == 5:
            self.play_dice_on_balloon()
            self.dice_reset()

        else:
            raise ValueError(f'Not determined action for {action_id} and for mask toplay: {to_play} and for number dice {self.num_dice} and for vector_action_id {vector_action_id}')


    def play_dice_on_balloon(self):

        for index, value in enumerate(self.states_dice):


            value = value.astype(int)


            if np.any(value == 0):
                pass
            else:
                # if state baloon not superior or equal to bust limit add column level



                if not self.states_balloons[0][value[0]-1] >= self.bust_limits_colors[value[0]-1]:

                    self.states_balloons[0][value[0]-1] += 1

                if not self.states_balloons[1][value[1]-1] >= self.bust_limits_shape[value[1]-1]:

                    self.states_balloons[1][value[1]-1] += 1

                if self.states_balloons[0][value[0]-1] >= self.bust_limits_colors[value[0]-1] \
                        or self.states_balloons[1][value[1]-1] >= self.bust_limits_shape[value[1]-1]:

                    self.num_breaks += 1
                    for index, value in enumerate(self.states_balloons):

                        value = value.astype(int)

                        for index2, value2 in enumerate(value):

                            if value2 <= self.bust_limits_all[index][index2]:
                                self.total_score += self.BALLOON_SCORES[index][index2][value2 - 1]


                            else:
                                raise ValueError(
                                    f'Balloon {index} is above busted level {value2} out of {self.bust_limits_colors[index]} ')

                    self.list_total_score = np.append(self.list_total_score, self.total_score)
                    self.dice_reset()
                    return None


    def dice_reset(self):
        self.states_dice = np.zeros((5,2), dtype=int)
        self.states_dice[:3, :2] = np.random.randint(1, 4, size=(3, 2))
        self.num_dice = 3



    def available_actions_ids(self) -> list:


        #[[0,0,


        if self.num_dice <= 4:
            return range(len(self.number_action_for4dice))

        else :
            raise ValueError(f'number dice false {self.num_dice} ')


    def get_real_score(self):

        return self.total_score




    def score(self) -> float:


        #tanh between 120 (-1 below) and 200 (1 above)
        # other_scoring = np.tanh((1/40)*(self.total_score - 160))

        # return other_scoring


        # return total score
        return self.total_score
    #     for index, value in enumerate(self.states_balloons):

    def dice_mask(self):

        if self.num_dice == 3:
            return np.array([1, 1, 1, 0])
        elif self.num_dice == 4:
            return np.array([1, 1, 1, 1])
        else:
            raise ValueError(f'number dice false {self.num_dice} ')

    def available_actions_mask(self) -> np.array:
        if self.num_dice == 3:

            return np.array([1] * 8 + [0] * 8)
        elif self.num_dice == 4:
            return np.array([1] * 16)
        else:
            raise ValueError(f'number dice false {self.num_dice} ')

    ##TODO: 3. Implement the reset_with_states function

    ##TODO: 4. Implement the view function

    ##TODO: 5. Implement the reset_random function






    def reset(self):
        self.states_balloons = np.zeros((2,3), dtype=int)
        self.dice_reset()
        self.num_breaks = 0
        self.total_score = 0
        self.list_total_score = np.array([])
        self.scoring = 0







# Play the game
# play_game()
# play_game_agent_DRL()

# env = BalloonPOPEnv()
#
# print(list(env.BUST_LIMITS.values())[0])
# print(env.number_action_for4dice)
# print(env.states_dice)
# print(env.state_vector())
# print(env.num_dice)
# print(env.act_with_action_id([1, 0, 1, 1, 0]))
# print(env.states_dice)
# print(env.state_vector())
# print(env.num_dice)
