import time
import random

from typing import Literal
from board import Board
from copy import deepcopy
from engines.engine import IEngine

MAX_VALUE = 100000


class MinimaxEngine(IEngine):
    def __init__(
        self,
        max_depth: int,
        color: Literal["@", "O"],
        current_board: Board,
        timer: bool = False,
        debugger: bool = False,
    ):
        self.max_depth = max_depth
        self.color = color
        self.opponent_color = "O" if color == "@" else "@"
        self.gain = -MAX_VALUE
        self.board = current_board
        self.timer = timer
        self.debugger = debugger
        print(f"Minimax engine discs: {self.color}\n")

    def perform_move(self):
        start = time.time()
        move = self.minimax(deepcopy(self.board), self.max_depth, self.color)
        end = time.time()

        if self.timer:
            print(
                f"Minimax engine move: (new disc position, discs to flip) = {move}, time: {end - start}"
            )

        return move

    def minimax(self, board: Board, depth: int, move_color: Literal["@", "O"]):
        # determine very high/low initial gain value
        gain = -MAX_VALUE if move_color == self.color else MAX_VALUE
        best_move = None
        # iterate all possible moves and create the game tree of given depth
        for move in board.find_possible_moves_for_player(move_color):
            new_board = deepcopy(board)
            new_board.update_board(move_color, move)

            # after going through the whole tree check current_gain for passed path
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
                    if current_gain <= gain:
                        gain = current_gain
                        best_move = move

                if self.debugger:
                    side = "MAX" if move_color == self.color else "MIN"
                    print(
                        f"{side}  Depth: {self.max_depth - depth}, move: {move}, gain: {current_gain}"
                    )

                self.minimax(new_board, depth - 1, "O" if move_color == "@" else "@")
            else:
                return move

        return best_move