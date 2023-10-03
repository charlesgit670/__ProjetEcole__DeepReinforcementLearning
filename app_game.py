# Constants
import random
from collections import Counter




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
    """Flatten a list."""
    return [item for sublist in l for item in sublist]

def roll_dice(number_of_dice=NUM_DICE):
    """Roll the dice and return a list of balloon colors."""
    return [random.choice(list(DICE_FACE.values())) for i in range(number_of_dice)]

def reroll_dice(dice, dice_to_reroll):
    """Reroll specific dice."""
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

def calculate_score(balloons_collected):
    """Calculate the score for the current turn."""
    score = 0
    for balloon, count in balloons_collected.items():
        if count <= len(BALLOON_SCORES[balloon]):
            score += BALLOON_SCORES[balloon][count - 1]
    return score

def simulated_input(choices):
    """Simulate a player's choice for demonstration purposes."""
    return random.choice(choices)

def play_turn():
    """Play one turn of the game."""
    balloons_collected = {color: 0 for color in BALLOONS}
    number_of_dice = NUM_DICE
    turn_score = 0

    while True:
        rolled = roll_dice(number_of_dice)
        print(f"You rolled: {rolled}")

        # Update the balloon collection based on the roll
        c = Counter(flatten(rolled))
        for balloon, count in c.items():
            balloons_collected[balloon] += count

        print(f"Balloons collected so far: {balloons_collected}")

        # Check for bust
        if is_busted(balloons_collected):
            print("Busted!")
            return 0

        # Calculate the score
        turn_score = calculate_score(balloons_collected)
        print(f"Current turn score: {turn_score}")

        # Simulate the player's choice to reroll or end their turn
        choice = simulated_input(['r', 'e'])
        print(f"Simulated choice: {choice}")
        if choice == 'e':
            return turn_score
        elif choice == 'r':
            dice_to_reroll = [random.randint(0, number_of_dice - 1) for _ in range(random.randint(1, number_of_dice))]
            print(f"Simulated dice to reroll: {dice_to_reroll}")
            rolled = reroll_dice(rolled, dice_to_reroll)

def play_game():
    """Play the Balloon Pop! game."""
    total_score = 0
    for turn in range(NUM_BREAKS):
        print(f"\n--- Turn {turn + 1} ---")
        total_score += play_turn()
        print(f"Total Score after turn {turn + 1}: {total_score}")

    print(f"\nGame Over! Your final score is: {total_score}")

# Play the game
play_game()

