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

def roll_dice(number_of_dice=NUM_DICE):
    """Roll the dice and return a list of balloon colors."""
    return [random.choice(list(DICE_FACE.values())) for i in range(number_of_dice)]

def reroll_dice(dice, dice_to_reroll):
    """Reroll specified dice."""
    nb_dice_to_reroll = len(dice_to_reroll) + 1
    reroll = roll_dice(nb_dice_to_reroll)
    for index, new_result in (zip(dice_to_reroll, range(nb_dice_to_reroll))):
        dice[index] = reroll[new_result]
    dice[len(dice)] = reroll[-1]
    return dice

def is_busted(balloons_collected):
    """Check if the player has busted."""
    for color, count in balloons_collected.items():
        if count >= BUST_LIMITS[color]:
            return True
    return False

def auto_reroll_dice(dice):
    """Automatically decide which dice to reroll for the bot."""
    return reroll_dice(dice, [0])

def auto_play_turn():
    """Play one turn of the game automatically without user input."""
    balloons_collected = {color: 0 for color in BALLOONS}
    number_of_dice = 3
    dice = {}
    rolled = roll_dice(number_of_dice)
    for index in range(number_of_dice):
        dice[index] = rolled[index]
    dice = auto_reroll_dice(dice)
    c = Counter(x for xs in rolled for x in set(xs))
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

def auto_play_game():
    """Automatically play the Balloon Pop! game without user input."""
    total_score = {color: 0 for color in BALLOONS}
    current_score = 0
    for turn in range(NUM_BREAKS):
        while not is_busted(total_score):
            balloons_collected = auto_play_turn()
            total_score = calculate_score(total_score, balloons_collected)
        current_score += count_score(total_score)
        total_score = {color: 0 for color in BALLOONS}
    return current_score

def benchmark():
    """Benchmark the game to determine how many games can be played in one second."""
    start_time = time.time()
    count = 0
    while time.time() - start_time < 1.0:
        auto_play_game()
        count += 1
    return count

# Benchmark the game
games_played_in_one_second = benchmark()
print(games_played_in_one_second)
