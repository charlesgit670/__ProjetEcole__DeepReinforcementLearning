import random
from collections import Counter
import time
import copy




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
    #
    return [item for sublist in l for item in sublist]
def roll_dice(number_of_dice=NUM_DICE, headless=False):
    """Roll the dice and return a list of balloon colors."""


    if not headless:

        return [random.choice(list(DICE_FACE.values())) for _ in range(number_of_dice)]

    else:

        rolled = [random.choice(list(DICE_FACE.keys())) for _ in range(number_of_dice)]
        return rolled


def reroll_dice(dice, dice_to_reroll, headless):


    print(dice_to_reroll)

    nb_dice_to_reroll = len(dice_to_reroll) + 1

    reroll = roll_dice(nb_dice_to_reroll, headless)

    for index, new_result in (zip(dice_to_reroll,range(nb_dice_to_reroll))):
        dice[index-1] = reroll[new_result]

    # for index, new_result in (zip(dice_to_reroll,range(nb_dice_to_reroll))):
    #     dice[index] = reroll[new_result]

    dice.append(reroll[-1])

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
    counts_color = {}
    counts_shape = {}

    list_color = []
    list_shape = []

    for values in d:
        color, shape = DICE_FACE[values]
        list_color.append(color)
        list_shape.append(shape)

    # set the unique values for color and shape to a dictionary
    counts_color = dict(Counter(list_color))
    counts_shape = dict(Counter(list_shape))

    counts = {**counts_color, **counts_shape}


    return counts


def play_turn(headless=False):
    """Play one turn of the game."""
    balloons_collected = {color: 0 for color in BALLOONS}
    number_of_dice = 3

    dice = {}


    rolled = roll_dice(number_of_dice, headless)

    rolled_to_return = copy.deepcopy(rolled)

    rolled_to_return += [0] * (NUM_DICE - number_of_dice)



    print(rolled)


    # for index in range(number_of_dice):
    #import copy

    #     dice[index] = rolled[index]
    print(f"You rolled: {rolled_to_return}")

    while len(rolled) < NUM_DICE:

        """return action agent DRL"""
        dice_to_reroll = [0,0,1]
        print(dice_to_reroll)
        dice_to_reroll = [int(i) for i in dice_to_reroll]


        rolled = reroll_dice(rolled, dice_to_reroll, headless)




        print(f"You rolled: {rolled}")





    c = count_elements(rolled)

    return c


def calculate_score(total_score, balloons_collected, headless=False):
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



def play_game_agent_DRL(headless=True):
    """Play the Balloon Pop! game."""
    total_score = {color: 0 for color in BALLOONS}

    print(total_score)
    current_score = 0
    # initialize the score


    for turn in range(NUM_BREAKS):

        print(f"\n--- Turn {turn + 1} ---")

        while not is_busted(total_score):
            print(f"Your accumulated balloons are {total_score}")
            print(f"The ultimate bust limits are {BUST_LIMITS}")
            balloons_collected = play_turn(headless)


            total_score = calculate_score(total_score, balloons_collected)


            #transform total score to list
            total_score_list = []
            for key, value in total_score.items():
                total_score_list.append(value)

            print(total_score_list)

        total_score = {color: 0 for color in BALLOONS}  # maybe it was reset balloons




import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0  # Starting value for epsilon in epsilon-greedy strategy
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum value for epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:
            return [random.randint(0, 1) for _ in range(5)]  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return [int(x) for x in torch.argmax(q_values, dim=1)]

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor(action)

        current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        max_next_q = self.model(next_state).max(1)[0]
        expected_q = reward + (0.99 * max_next_q * (1 - int(done)))

        loss = self.loss_fn(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




# Play the game
# play_game()
play_game_agent_DRL()


