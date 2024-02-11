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

        self.DICE_FACE = {
            1: [self.BALLOON_COLORS[2], self.BALLOON_SHAPES[0]],
            2: [self.BALLOON_COLORS[2], self.BALLOON_SHAPES[1]],
            3: [self.BALLOON_COLORS[2], self.BALLOON_SHAPES[2]],
            4: [self.BALLOON_COLORS[1], self.BALLOON_SHAPES[0]],
            5: [self.BALLOON_COLORS[1], self.BALLOON_SHAPES[1]],
            6: [self.BALLOON_COLORS[0], self.BALLOON_SHAPES[0]],
        }

        self.BALLOON_SCORES = {
            "Yellow": [0,3,7,11,15,3],
            "Blue": [1,3,5,7,9,12,8],
            "Red": [0,0,0,2,4,6,8,10,14,6],
            "Star": [1,2,3,5,7,10,13,16,4],
            "Moon": [2,3,4,5,7,9,12,5],
            "Kite": [1,3,6,10,13,7]
        }

        self.num_breaks = 0

        self.state_size = 16 #6 ballons [2, 1, 3, 1, 0, 4] et 5 dés [color [0 || 1 || 2 || 3], shape [0 || 1 || 2 || 3]] * 5
        self.action_size = 5 # si le joueur décide de relancer les dés 1, 3 et 4, le vecteur serait : [1, 0, 1, 1, 0]
        self.states_balloons = np.zeros((2,3))
        self.states_dice = np.zeros((5,2))
        self.num_dice = 3

        self.number_action_for3dice = self.generate_binary_numbers(3)
        self.number_action_for4dice = self.generate_binary_numbers(4)



        self.reset()

    def generate_binary_numbers(self, n):
        # Create an iterable of 0 and 1
        bits = [0, 1]
        # Use itertools.product to generate all combinations, specifying repeat=n
        binary_numbers = itertools.product(bits, repeat=n)
        # Convert each combination to a binary string and return the list
        initial_binary_numbers = [list(number) for number in binary_numbers]

        extended_binary_numbers = [number + [0] * (5 - len(number)) for number in initial_binary_numbers]

        return extended_binary_numbers

    def state_vector(self) -> np.array:
        state_vector = np.concatenate((self.states_balloons, self.states_dice), axis=None)
        return np.array(state_vector).flatten()

    def is_game_over(self) -> bool:

        if self.num_breaks >= self.NUM_MAX_BREAKS:
            return True

        return False

    def act_with_action_id(self, action_id): #action_id: [0, 1, 0, 1, 0]
        assert not self.is_game_over(), "Attempted to act in a finished game."


        to_play = action_id * available_actions_mask() # [1, 0, 1, 1, 1] * [1, 1, 1, 0, 0] = [1, 0, 1, 0, 0]

        if np.any(to_play == 1) and self.num_dice < 5:
            self.num_dice += 1

            for index, value in enumerate(to_play):

                if value == 1:
                    if index not in self.available_actions_ids():
                        # Properly format the error message
                        available_actions = self.available_actions_ids()
                        error_message = f'DICE {index} is not in the available actions: {available_actions}'
                        raise ValueError(error_message)

                    self.states_dice[index] = np.random.randint(1, 4)

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
            raise ValueError(f'Not determined action for {action_id} and for: {available_actions}')




    def play_dice_on_balloon(self):

        for index, value in enumerate(self.states_dice):
            if np.any(value == 0):
                pass
            else:

                self.states_balloons[0][value[0]] += 1
                self.states_balloons[1][value[1]] += 1

    def dice_reset(self):
        self.states_dice = np.zeros((5,2))
        self.num_dice = 3

    def available_actions_ids(self) -> list:

        if self.num_dice == 3:
            return self.number_action_for3dice
        elif self.num_dice == 4:
            return self.number_action_for4dice

        else :
            raise ValueError(f'number dice false {self.num_dice} ')



    ##TODO: 1. Implement the score function
    # def score(self) -> float:
    #
    #     for index, value in enumerate(self.states_balloons):

    ##TODO: 2. Implement the available_actions_mask function

    ##TODO: 3. Implement the reset_with_states function

    ##TODO: 4. Implement the view function

    ##TODO: 5. Implement the reset_random function






    def reset(self):
        self.states_balloons = np.zeros(6)
        self.states_dice = np.zeros((5,2))
        self.num_breaks = 0
        self.num_dice = 3


    def flatten(self, l):
            #
        return [item for sublist in l for item in sublist]
    def roll_dice(self, headless=False):
        """Roll the dice and return a list of balloon colors."""

        if not headless:

            return [random.choice(list(self.DICE_FACE.values())) for _ in range(self.NUM_DICE)]

        else:

            return [random.choice(list(self.DICE_FACE.keys())) for _ in range(self.NUM_DICE)]



    def reroll_dice(self, dice, dice_to_reroll):

        nb_dice_to_reroll = len(dice_to_reroll) + 1
        reroll = roll_dice(nb_dice_to_reroll)

        for index, new_result in (zip(dice_to_reroll,range(nb_dice_to_reroll))):
            dice[index] = reroll[new_result]

        dice[len(dice)] = reroll[-1]
        return dice

    def is_busted(self, balloons_collected):
        """Check if the player has busted."""
        for color, count in balloons_collected.items():

            if count >= BUST_LIMITS[color]:

                print(f"Ooops! You busted! {color} exploded!")

                return True

        return False


    def count_elements(self, d):
        # Initialize an empty dictionary to store the counts
        counts = {}

        # Loop through the values of the input dictionary
        for value_list in d.values():
            for value in value_list:
                # If the value is already in the counts dictionary, increment its count
                if value in counts:
                    counts[value] += 1
                # Otherwise, add the value to the dictionary with a count of 1
                else:
                    counts[value] = 1

        return counts


    def play_turn(self, headless=False):
        """Play one turn of the game."""
        balloons_collected = {color: 0 for color in BALLOONS}
        number_of_dice = 3

        dice = {}


        rolled = roll_dice(number_of_dice, headless)

        print(rolled)


        for index in range(number_of_dice):

            dice[index] = rolled[index]
        print(f"You rolled: {dice}")

        while len(dice) < NUM_DICE:

            reroll = input("Do you want to reroll? (y/n) ")
            if reroll == "y":

                dice_to_reroll = input("Which dice do you want to reroll? (e.g. 0 1 2 3 4) ")



                dice_to_reroll = [int(i) for i in dice_to_reroll.split()]
                dice = reroll_dice(dice, dice_to_reroll)
                print(dice)


            else:


                break


        c = count_elements(dice)
        print(c)
        print("c")

        return c


    def calculate_score(self, total_score, balloons_collected, headless=False):
        """Calculate the score for the current turn."""

        for balloon, count in balloons_collected.items():
            total_score[balloon] += count
        return total_score

    def count_score(self, total_score):
        """Calculate the score for the current turn."""
        scoring = 0
        for balloon, count in total_score.items():
            if total_score[balloon] <= BUST_LIMITS[balloon]:
                scoring += BALLOON_SCORES[balloon][total_score[balloon]-1]
            else:
                scoring += 0


        return scoring



    def play_game(self, headless=False):
        """Play the Balloon Pop! game."""
        total_score = {color: 0 for color in BALLOONS}
        current_score = 0
        print("Welcome to Balloon Pop!\n")
        print("Your score is", current_score)

        for self.num_breaks in range(self.NUM_MAX_BREAKS):
            print(f"\n--- Turn {turn + 1} ---")

            while not is_busted(total_score):
                print(f"Your accumulated balloons are {total_score}")
                print(f"The ultimate bust limits are {BUST_LIMITS}")
                balloons_collected = play_turn()
                total_score = calculate_score(total_score, balloons_collected)
                print('total_score')
                print(total_score)

            print(f"Your accumulated balloons are {total_score}")
            current_score += count_score(total_score)
            print(f"Your current accumulated score is {current_score}")
            total_score = {color: 0 for color in BALLOONS} #maybe it was reset balloons


    def play_game_agent_DRL(self, headless=True):
        """Play the Balloon Pop! game."""
        total_score = {color: 0 for color in BALLOONS}
        current_score = 0
        # initialize the score


        for num_breaks in range(self.NUM_MAX_BREAKS):

            print(f"\n--- Turn {turn + 1} ---")

            while not is_busted(total_score):
                print(f"Your accumulated balloons are {total_score}")
                print(f"The ultimate bust limits are {BUST_LIMITS}")
                balloons_collected = play_turn(headless)
                total_score = calculate_score(total_score, balloons_collected)







# Play the game
# play_game()
# play_game_agent_DRL()

env = BalloonPOPEnv()

print(env.number_action_for3dice)
print(env.states_dice)
print(env.state_vector())
print(env.num_dice)
print(env.act_with_action_id([1, 0, 1, 1, 0]))
print(env.states_dice)
print(env.state_vector())
print(env.num_dice)
