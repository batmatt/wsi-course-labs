from engines.engine import IEngine


class MinimaxEngine(IEngine):
    def __init__(self, depth, is_maximizing_player, color):
        self.depth = depth
        self.is_maximizing_player = is_maximizing_player
        self.color = color
        print(f"Minimax engine color: {self.color}\n")

    def perform_move(self, possible_moves):
        if self.is_maximizing_player:
            pass
