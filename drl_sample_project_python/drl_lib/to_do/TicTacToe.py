from ..do_not_touch.contracts import SingleAgentEnv
import numpy as np

# player_id = 0 if player
# else it is 1 if bot

"""
L'action i correspond à cocher la case i. Les cases sont numérotées comme suit:
[
	[0, 1, 2],
	[3, 4, 5],
	[6, 7, 8]
]
"""


class TicTacToe(SingleAgentEnv):
    def __init__(self):
        self.agent_pos = 0
        self.game_over = False
        self.current_score = 0.0
        self.marks = {0: 'X', 1: 'O'}
        self.board = np.array(3 * [3 * [' ']], np.str)
        self.winner = -1

    def reset(self):
        self.agent_pos = 0
        self.game_over = False
        self.current_score = 0.0
        self.board = np.array(3 * [3 * [' ']], np.str)
        self.winner = -1

    def state_id(self) -> int:
        return self.agent_pos

    def is_game_over(self) -> bool:
        return self.game_over

    def get_coord(self, action):
        row = int(action / 3)
        column = int(action % 3)
        return row, column

    def act_with_action_id(self, action_id: int):
        assert (0 <= action_id < 9)
        assert (not self.game_over)

        row, column = self.get_coord(action_id)

        if self.board[row][column] not in self.marks.values():
            self.board[row][column] = self.marks[0]
            if self.is_terminal(self.marks[0]):
                self.game_over = True
                self.current_score = 1.0
                self.winner = 0
            self.agent_pos = action_id


        else:
            action = np.random.choice(self.available_actions_ids())
            row, column = self.get_coord(action)

            if self.board[row][column] not in self.marks.values():
                self.board[row][column] = self.marks[1]
                if self.is_terminal(self.marks[0]):
                    self.game_over = True
                    self.winner = 1
                self.agent_pos = action

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        values = np.array([], np.int32)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.board[i][j] not in self.marks.values():
                    values = np.append(i * 3 + j, values)
        return np.array(values)

    def check_row_win(self, board, player):
        for row in board:
            if len(set(row)) == 1 and row[0] == player:
                return True
        return False

    def check_column_win(self, board, player):
        return self.check_row_win(np.transpose(board), player)

    def check_diagonals_win(self, board, player):
        if len(set([board[i][i] for i in range(len(board))])) == 1 and board[0][0] == player:
            return True
        if len(set([board[i][len(board) - i - 1] for i in range(len(board))])) == 1 and board[2][0] == player:
            return True
        return False

    def is_successfull(self, player):
        return self.check_row_win(self.board, player) or \
               self.check_column_win(self.board, player) or \
               self.check_diagonals_win(self.board, player)

    def is_terminal(self, player):
        return self.is_successfull(player) or len(self.available_actions_ids()) == 0
