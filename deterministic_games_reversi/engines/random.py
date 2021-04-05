import random

from engines.engine import IEngine


class RandomEngine(IEngine):
    def __init__(self, color) -> None:
        self.color = color
        print(f"Random engine color: {self.color}\n")

    def perform_move(self, possible_moves):
        moves_list = list(possible_moves.items())
        move = random.choice(moves_list)

        return move
