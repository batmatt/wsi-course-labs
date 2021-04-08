"""
Demonstration of Reversi/Othello using Minimax engine
by
Mateusz Winnicki
"""

import argparse
import time
from board import Board, BLACK, WHITE
from engines.minimax import MinimaxEngine
from engines.random import RandomEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--board-size", type=int, help="Length of the board's side (must be even)"
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        help="Depth of minimax algorithm (must be larger than 1)",
    )
    parser.add_argument(
        "-be",
        "--black_engine",
        type=str,
        help="Determines which engine should play Black. Possible options: Random, Minimax - [rd | mm]",
    )
    parser.add_argument(
        "-we",
        "--white_engine",
        type=str,
        help="Determines which engine should play White. Possible options: Random, Minimax - [rd | mm]",
    )
    parser.add_argument(
        "-t",
        "--timer",
        action="store_true",
        help="Displays time of making a move by engines",
    )
    parser.add_argument(
        "-db",
        "--debugger",
        action="store_true",
        help="Enables debug option displaying game trees built by minimax",
    )

    args = parser.parse_args()

    _board_size = args.board_size
    _depth = args.depth
    _black_engine = args.black_engine
    _white_engine = args.white_engine
    _timer = args.timer
    _debugger = args.debugger

    if _depth < 1:
        print("Wrong depth of minimax algorithm (must be larger than 0)")
        exit(1)

    if _board_size % 2 == 0 and _board_size >= 2:
        print("Initial board\n")
        _board = Board(_board_size)
        _board.print_board()

        first_player = None
        second_player = None

        if _black_engine == "rd":
            first_player = RandomEngine("@", _board, _timer)
        elif _black_engine == "mm":
            first_player = MinimaxEngine(_depth, "@", _board, _timer, _debugger)
        else:
            print("Could not find given engine for Black")
            exit(1)

        if _white_engine == "rd":
            second_player = RandomEngine("O", _board, _timer)
        elif _white_engine == "mm":
            second_player = MinimaxEngine(_depth, "O", _board, _timer, _debugger)
        else:
            print("Could not find given engine for White")
            exit(1)

        player_turn = BLACK

        # make moves based on engine calculations until one of the players can't place another disc in its turn
        _turns = 0

        start = time.time()
        while not _board.is_state_terminal(player_turn, _turns):
            _turns = _turns + 1
            if player_turn == BLACK:
                _board.update_board(
                    player_turn,
                    first_player.perform_move(),
                )
                _board.print_board()
                player_turn = WHITE
            else:
                _board.update_board(
                    player_turn,
                    second_player.perform_move(),
                )
                _board.print_board()
                player_turn = BLACK

        end = time.time()
        if _timer:
            print(f"Whole game took {end - start} seconds")

    else:
        print("Wrong length of the board's side (must be even)")
        exit(1)


if __name__ == "__main__":
    main()
