import argparse
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
        "-sa",
        "--starting-engine",
        type=str,
        help="Determines which engine should move first (play black). Possible options: Random, Minimax - [rd | mm]",
    )

    args = parser.parse_args()

    _board_size = args.board_size
    _depth = args.depth
    _starting_engine = args.starting_engine

    if _depth < 1:
        print("Wrong depth of minimax algorithm (must be larger than 1)")
        exit(1)

    if _board_size % 2 == 0 and _board_size >= 2:
        print("Initial board\n")
        _board = Board(_board_size)
        _board.print_board()

        first_player = RandomEngine("Black")
        second_player = MinimaxEngine(_depth, False, "White")

        if _starting_engine == "mm":
            first_player = MinimaxEngine(_depth, True, "Black")
            second_player = RandomEngine("White")

        player_turn = BLACK

        while not _board.is_state_terminal(player_turn):
            if player_turn == BLACK:
                _board.update_board(
                    player_turn,
                    first_player.perform_move(
                        _board.find_possible_moves_for_player(player_turn)
                    ),
                )
                player_turn = WHITE
            else:
                _board.update_board(
                    player_turn,
                    second_player.perform_move(
                        _board.find_possible_moves_for_player(player_turn)
                    ),
                )
                player_turn = BLACK

    else:
        print("Wrong length of the board's side (must be even)")
        exit(1)


if __name__ == "__main__":
    main()
