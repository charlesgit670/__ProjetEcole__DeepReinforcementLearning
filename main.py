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
    1 : [BALLOON_COLORS[2], BALLOON_SHAPES[0]],
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

    for index in dice_to_reroll:
        dice[index] = reroll

    return dice


def is_busted(balloons_collected):
    """Check if the player has busted."""
    for color, count in balloons_collected.items():
        if count > BUST_LIMITS[color]:
            return True
    return False


def play_turn():
    """Play one turn of the game."""
    balloons_collected = {color: 0 for color in BALLOONS}
    number_of_dice = 3

    dice = {}

    while True:
        rolled = roll_dice(number_of_dice)
        print(f"You rolled: {rolled}")

        for index in range(number_of_dice):

            dice[index] = rolled[index]
        print(dice)
        print(len(dice))

        c = Counter(x for xs in rolled for x in set(xs))
        print(c)



def play_game():
    """Play the Balloon Pop! game."""
    # total_score = 0
    # for turn in range(NUM_BREAKS):
    #     print(f"\n--- Turn {turn + 1} ---")
    #     total_score += play_turn()
    #     print(f"Total Score: {total_score}")
    #
    # print(f"\nGame Over! Your final score is: {total_score}")
    # return total_score
    #
    # """other test"""
    play_turn()



    # start_time = time.time()
    # pip = []
    # for i in range(2):
    #     pip.append(roll_dice())
    #
    # print(pip)

    # c = Counter(element for sublist in pip for pair in sublist for element in pair)
    # print(c)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))




# Play the game
# play_game()
