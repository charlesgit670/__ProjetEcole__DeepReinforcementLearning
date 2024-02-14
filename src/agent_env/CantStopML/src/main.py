import random
import time
import numpy as np
import copy


class CantStopGame:
    def __init__(self, logs: bool = True):
        self.board = self.initialize_board()
        self.current_player = 0
        self.blocked_columns = [0, 0]
        # self.active_columns = set()
        self.last_dice_roll = []
        self.is_over = False
        self.logs = logs

    def roll_dice(self):

        return [random.randint(1, 6) for _ in range(4)]

    def get_combinations(self, dice_roll):

        return [
            (dice_roll[0] + dice_roll[1], dice_roll[2] + dice_roll[3]),
            (dice_roll[0] + dice_roll[2], dice_roll[1] + dice_roll[3]),
            (dice_roll[0] + dice_roll[3], dice_roll[1] + dice_roll[2])
        ]

    def initialize_board(self):

        column_lengths = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}

        self.active_columns = set()

        return {column: {"length": length, "progress": [0, 0], "blocked": False} for column, length in
                column_lengths.items()}

    def observe(self):

        # Extract progress values
        progress_values = [self.board[col]['progress'] for col in self.board]

        # Convert into numpy arrays and split in two sub arrays
        progress_arrays = np.array(progress_values).T  # Transpose to split values
        return progress_arrays

    def show_board_status(self):

        board_status_parts = []
        for col in self.board:
            col_status = f"Column {col}: {self.board[col]['progress'][self.current_player]}/{self.board[col]['length']} {'(Blocked)' if self.board[col]['blocked'] else ''}"
            board_status_parts.append(col_status)
        board_status = ', '.join(board_status_parts)
        if self.logs: print(f"Player {self.current_player + 1} Board Status: [{board_status}]")

    def possible_actions(self):

        self.last_dice_roll = self.roll_dice()
        if self.logs: print("Dices rolled:", self.last_dice_roll)
        combinations = self.get_combinations(self.last_dice_roll)

        valid_combinations = []
        for comb in combinations:
            if not self.board[comb[0]]['blocked'] and not self.board[comb[1]]['blocked']:
                valid_combinations.append(comb)

        if self.logs: print("Valid combinations:", valid_combinations)
        return valid_combinations

    def step(self, action):

        reward = 0

        columns_to_advance = action
        if self.logs: print("Selected columns:", columns_to_advance)

        for column in columns_to_advance:
            if len(self.active_columns) < 3 or column in self.active_columns:
                if self.board[column]["progress"][self.current_player] < self.board[column]["length"] and not \
                self.board[column]["blocked"]:
                    self.board[column]["progress"][self.current_player] += 1  # Advance
                    self.active_columns.add(column)
            else:
                if self.logs: print(f"Cannot advance in column {column} as already 3 columns are active.")
                continue

        for col in self.board:
            if self.board[col]["progress"][self.current_player] >= self.board[col]["length"] and not self.board[col]["blocked"]:
                self.board[col]["blocked"] = True
                self.blocked_columns[self.current_player] += 1
                if self.logs: print(f"Player {self.current_player + 1} has blocked column {col}.")
                reward += 1

        if self.blocked_columns[self.current_player] == 3:
            self.is_over = True

            if self.current_player == 0:
                if self.logs: print(f"Win. Player {self.current_player + 1} is the winner")
                reward += 2
            else:

                if self.logs: print(f"Loose. Player {self.current_player + 1} is the winner")
                reward = -2

            return reward

        self.show_board_status()
        return reward

    def play(self):

        start_time = time.perf_counter()

        turn_number = 0

        while not self.is_over:
            if self.logs: print(f"\nPlayer {self.current_player + 1}'s turn")

            # Model Turn
            if self.current_player == 0:

                if self.logs: print("IA turn", turn_number)

                if turn_number == 0:
                    saved_board = copy.deepcopy(self.board)
                    if self.logs: print("Following board saved : "), self.show_board_status()

                self.active_columns = set()

                valid_actions = self.possible_actions()

                if not valid_actions:
                    if self.logs: print("No possible moves")

                    if turn_number > 0:
                        self.show_board_status()
                        if self.logs: print("No possibility after choised to contnue, board restored")
                        self.board = saved_board
                        self.show_board_status()

                    self.current_player = 1 - self.current_player
                    continue

                # Select a random action
                action = random.choice(valid_actions)

                reward = self.step(action)
                if self.logs: print("reward : ", reward)

                # Continue or stop
                if random.choice([True, False]):
                    if self.logs: print("Player decides to stop.")
                    self.current_player = 1 - self.current_player
                    turn_number = 0

                else:
                    if self.logs: print("Player decides to continue.")
                    turn_number += 1

            # Random turn
            else:

                if self.logs: print("Random turn")

                if turn_number == 0:
                    saved_board = copy.deepcopy(self.board)
                    if self.logs: print("Following board saved : "), self.show_board_status()

                self.active_columns = set()

                valid_actions = self.possible_actions()

                if not valid_actions:
                    if self.logs: print("No possible moves")

                    if turn_number > 0:
                        self.show_board_status()
                        if self.logs: print("No possibility after choised to contnue, board restored")
                        self.board = saved_board
                        self.show_board_status()

                    self.current_player = 1 - self.current_player
                    continue

                # Select a random action
                action = random.choice(valid_actions)

                reward = self.step(action)
                if self.logs: print("reward : ", reward)

                # continue or not
                if random.choice([True, False]):
                    if self.logs: print("Player decides to stop.")
                    self.current_player = 1 - self.current_player
                    turn_number = 0
                else:
                    if self.logs: print("Player decides to continue.")
                    turn_number += 1

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Total game time: {duration:.6f} seconds.")


game = CantStopGame(logs=True)
game.play()