import random
import time
from typing import Literal
from board import Board

from engines.engine import IEngine


class RandomEngine(IEngine):
    """
    Engine always choosing random moves from all possible

    """

    def __init__(
        self,
        color: Literal["@", "O"],
        current_board: Board,
        timer: bool = False,
    ) -> None:
        self.color = color
        self.board = current_board
        self.timer = timer
        print(f"Random engine discs: {self.color}\n")

    def perform_move(self):
        possible_moves = self.board.find_possible_moves_for_player(self.color)
        start = time.time()
        move = random.choice(possible_moves)
        end = time.time()

        if self.timer:
            print(
                f"Random engine move: (new disc position, discs to flip) = {move}, time: {end - start}"
            )

        return move
