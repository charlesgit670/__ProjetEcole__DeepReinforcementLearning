import random
from collections import Counter
import time

# Constants
BALLOON_COLORS = ["Yellow", "Blue", "Red"]
BALLOON_SHAPES = ["Star", "Moon", "Kite"]

BALLOONS = BALLOON_COLORS + BALLOON_SHAPES

BUST_LIMITS = {
    "Yellow": 6,
    "Blue": 7,
    "Red": 10,
    "Star": 9,
    "Moon": 8,
    "Kite": 6
}
DICE_FACE = {
    1: [BALLOON_COLORS[2], BALLOON_SHAPES[0]],
    2: [BALLOON_COLORS[2], BALLOON_SHAPES[1]],
    3: [BALLOON_COLORS[2], BALLOON_SHAPES[2]],
    4: [BALLOON_COLORS[1], BALLOON_SHAPES[0]],
    5: [BALLOON_COLORS[1], BALLOON_SHAPES[1]],
    6: [BALLOON_COLORS[0], BALLOON_SHAPES[0]],
}

BALLOON_SCORES = {
    "Yellow": [0,3,7,11,15,3],
    "Blue": [1,3,5,7,9,12,8],
    "Red": [0,0,0,2,4,6,8,10,14,6],
    "Star": [1,2,3,5,7,10,13,16,4],
    "Moon": [2,3,4,5,7,9,12,5],
    "Kite": [1,3,6,10,13,7]
}

NUM_DICE = 5
NUM_BREAKS = 3

def flatten(l):
    return [item for sublist in l for item in sublist]
def roll_dice(number_of_dice=NUM_DICE):
    """Roll the dice and return a list of balloon colors."""


    return [random.choice(list(DICE_FACE.values())) for i in range(number_of_dice)]


def reroll_dice(dice, dice_to_reroll):

    nb_dice_to_reroll = len(dice_to_reroll) + 1

    reroll = roll_dice(nb_dice_to_reroll)

    for index, new_result in (zip(dice_to_reroll,range(nb_dice_to_reroll))):
        dice[index] = reroll[new_result]

    dice[len(dice)] = reroll[-1]

    return dice


def is_busted(balloons_collected):
    """Check if the player has busted."""
    for color, count in balloons_collected.items():

        if count >= BUST_LIMITS[color]:

            print(f"Ooops! You busted! {color} exploded!")

            return True

    return False


def count_elements(d):
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


def play_turn():
    """Play one turn of the game."""
    balloons_collected = {color: 0 for color in BALLOONS}
    number_of_dice = 3

    dice = {}


    rolled = roll_dice(number_of_dice)


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

    return c


def calculate_score(total_score, balloons_collected):
    """Calculate the score for the current turn."""

    for balloon, count in balloons_collected.items():
        total_score[balloon] += count
    return total_score

def count_score(total_score):
    """Calculate the score for the current turn."""
    scoring = 0
    for balloon, count in total_score.items():
        if total_score[balloon] <= BUST_LIMITS[balloon]:
            scoring += BALLOON_SCORES[balloon][total_score[balloon]-1]
        else:
            scoring += 0
    return scoring



def play_game():
    """Play the Balloon Pop! game."""
    total_score = {color: 0 for color in BALLOONS}
    current_score = 0
    print("Welcome to Balloon Pop!\n")
    print("Your score is", current_score)

    for turn in range(NUM_BREAKS):

        print(f"\n--- Turn {turn + 1} ---")

        while not is_busted(total_score):
            print(f"Your accumulated balloons are {total_score}")
            print(f"The ultimate bust limits are {BUST_LIMITS}")
            balloons_collected = play_turn()
            total_score = calculate_score(total_score, balloons_collected)


        print(f"Your accumulated balloons are {total_score}")
        current_score += count_score(total_score)
        print(f"Your current accumulated score is {current_score}")
        total_score = {color: 0 for color in BALLOONS}



# Play the game
play_game()
