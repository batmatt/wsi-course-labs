import random

from engines.engine import IEngine


class RandomEngine(IEngine):
    """
    Engine always choosing random moves from all possible

    """

    def __init__(self, color: str) -> None:
        self.color = color
        print(f"Random engine color: {self.color}\n")

    def perform_move(self, possible_moves: dict):
        moves_list = list(possible_moves.items())
        move = random.choice(moves_list)

        return move
