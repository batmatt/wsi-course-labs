from typing import Literal


COORD_X = 0
COORD_Y = 1

BLACK = "@"
WHITE = "O"
EMPTY = "."

DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


class Board:
    """
    Class representing the board state

    """

    def __init__(self, size: int) -> None:
        self.size = size

        # Fill the board with empty places
        self.board = [[EMPTY for x in range(self.size)] for y in range(self.size)]

        # Initialize starting discs in the centre of the board
        half = (self.size) // 2
        self.board[half - 1][half] = BLACK
        self.board[half][half - 1] = BLACK
        self.board[half - 1][half - 1] = WHITE
        self.board[half][half] = WHITE

    def print_board(self):
        for row in range(len(self.board)):
            print(*self.board[row], sep=" ")
        print("\n")

    def update_board(self, player_turn: Literal["@", "O"], move) -> None:
        """
        Updates board after player makes a move in its turn

        """

        new_disc_position = move[0]
        discs_to_flip = move[1]

        if player_turn == BLACK:
            self.board[new_disc_position[COORD_X]][new_disc_position[COORD_Y]] = BLACK
            for flip in discs_to_flip:
                self.board[flip[COORD_X]][flip[COORD_Y]] = BLACK

        elif player_turn == WHITE:
            self.board[new_disc_position[COORD_X]][new_disc_position[COORD_Y]] = WHITE
            for flip in discs_to_flip:
                self.board[flip[COORD_X]][flip[COORD_Y]] = WHITE

    def count_discs(self):
        """
        Returns current state of black and white's discs count

        """
        white_count = 0
        black_count = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == WHITE:
                    white_count += 1
                if self.board[x][y] == BLACK:
                    black_count += 1

        return white_count, black_count

    def get_all_discs_coords(self, color: Literal["@", "O"]) -> list:
        """
        Returns a list with positions of each disc of given color

        """
        discs = []

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == color:
                    discs.append((x, y))

        return discs

    def find_possible_moves_for_player(self, color: Literal["@", "O"]) -> list:
        """
        Finds all possible moves for each disc of choosen color.
        Returns dict with values representing positions of opposing color discs
        which will be flipped after making a legal move.

        """
        possible_moves = {}

        if color == WHITE:
            opposing_color = BLACK
        else:
            opposing_color = WHITE

        player_discs = Board.get_all_discs_coords(self, color)

        for disc in player_discs:
            for direction in DIRECTIONS:
                flips = []
                for i in range(self.size):
                    dx = disc[COORD_X] + direction[COORD_X] * (i + 1)
                    dy = disc[COORD_Y] + direction[COORD_Y] * (i + 1)

                    if dx < 0 or dy < 0 or dx > self.size - 1 or dy > self.size - 1:
                        break
                    elif self.board[dx][dy] == opposing_color:
                        flips.append((dx, dy))
                    # if we step into color in line after finding possible flips of opposing color we must drop them ;/ that's the rules, i don't make the rules
                    elif self.board[dx][dy] == color and flips:
                        flips = []
                    elif self.board[dx][dy] == EMPTY and flips:
                        possible_moves[(dx, dy)] = flips
                        # break to stop on the first empty space after opposing color discs
                        break

        return list(possible_moves.items())

    def is_state_terminal(self, player_turn: Literal["@", "O"], turns: int):
        # current player have to have at least one disc on the board
        if self.get_all_discs_coords(player_turn):
            # current player have to be able to perform some move
            if self.find_possible_moves_for_player(player_turn):
                return False
            else:
                w, b = self.count_discs()
                print(
                    f"\nGame finished after {turns} turns.\nBlack score: {b}\nWhite score: {w}"
                )
                return True
        else:
            w, b = self.count_discs()
            print(f"\nGame finished {turns} turns.\nBlack score: {b}\nWhite score: {w}")
            return True
