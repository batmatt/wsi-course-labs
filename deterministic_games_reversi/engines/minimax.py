from typing import Literal
from board import Board
from copy import deepcopy
from engines.engine import IEngine

MAX_VALUE = 100000


class MinimaxEngine(IEngine):
    def __init__(self, max_depth: int, color: Literal["@", "O"], current_board: Board):
        self.max_depth = max_depth
        self.color = color
        self.opponent_color = "O" if color == "@" else "@"
        self.gain = -MAX_VALUE
        self.board = current_board
        print(f"Minimax engine color: {self.color}\n")

    def perform_move(self):
        return self.minimax(deepcopy(self.board), self.max_depth, self.color)

    def minimax(self, board: Board, depth: int, move_color: Literal["@", "O"]):
        gain = -MAX_VALUE if move_color == self.color else MAX_VALUE
        best_move = None
        for move in board.find_possible_moves_for_player(move_color):
            new_board = deepcopy(board)
            new_board.update_board(move_color, move)
            if depth > 0:
                w, b = new_board.count_discs()
                if self.color == "O":
                    current_gain = w - b
                else:
                    current_gain = b - w

                if move_color == self.color:
                    if current_gain > gain:
                        gain = current_gain
                        best_move = move
                else:
                    if current_gain < gain:
                        gain = current_gain
                        best_move = move

                self.minimax(new_board, depth - 1, "O" if move_color == "@" else "@")
            else:
                return move

        return best_move