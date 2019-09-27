import numpy as np
from alpha_zero_general.Game import Game
from Nine_Men_Morris_Alpha_2.Game.NMMLogic import Board
from numba import jit


def compress_tensor(tensor_board: np.ndarray) -> np.ndarray:
    """
    after optimize it takes 4 us to run
    Compress state representation to 7 by 7 board
    :param tensor_board:
    :return: 7 by 7 np.ndarray
    """
    assert tensor_board.shape == (7, 7, 7), \
        f"Invalid Input Shape: shape should be: (7,7,7) shape given was :{tensor_board.shape}"
    p1_layer = tensor_board[2, :, :]
    encoding_layer = tensor_board[3, :, :]
    p2_layer = tensor_board[4, :, :] * -1
    compressed_board = p1_layer + encoding_layer + p2_layer
    return compressed_board


def decompress_tensor(prev_tensor_board: np.ndarray, new_board: np.ndarray) -> np.ndarray:
    """
    De-Compress state representation to 3 by 7 by 7 tensor board and push the older states out
    after obtimize the times it takes 48 us per run still really big
    :param prev_tensor_board:
    :param new_board:
    :return: 7 by 7 by 7 np.ndarray
    """
    assert new_board.shape == (7, 7), "Invalid Input Shape "
    assert prev_tensor_board.shape == (7, 7, 7), "Invalid Input Shape "
    # this change made the time of the function to be half
    encoding_mask = np.array(
        [
            [0, 1, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 1, 0],
        ]
        )
    encoding_layer = encoding_mask * new_board
    p1_mask = np.zeros([7, 7])
    p2_mask = np.zeros([7, 7])
    p1_mask[np.where(new_board == 1)] = 1
    p2_mask[np.where(new_board == -1)] = -1  # mask also negates values for player 2
    p1_layer = new_board * p1_mask * (1 - encoding_mask)  # Todo Need to check if a float number will give a error
    p2_layer = new_board * p2_mask
    # push older layers mapping: new board goes to : [2,3,4] (p1,step,p2) ; t0: [2,4] -> t_1: [1,5], t1: [1,5] -> [0,6]
    new_tensor_board = np.zeros([7, 7, 7])
    layer_mappings = [(1, 0), (2, 1), (4, 5), (5, 6)]  # (from, to)
    for (from_, to_) in layer_mappings:
        new_tensor_board[to_, :, :] = prev_tensor_board[from_, :, :]
    new_tensor_board[2, :, :] = p1_layer
    new_tensor_board[3, :, :] = encoding_layer
    new_tensor_board[4, :, :] = p2_layer
    return new_tensor_board


def flip_tensor(tensor_board: np.ndarray) -> np.ndarray:
    """
    switch between player 1 and player -1
    :param tensor_board:
    :return: 7 by 7 by 7 np.ndarray
    """
    assert tensor_board.shape == (7, 7, 7), "Invalid Input Shape "
    return np.flip(tensor_board, 0)


class MenMorris(Game):
    def __init__(self, men_count):
        super(MenMorris, self).__init__()
        self.men_count = men_count
        self.actionSize = 24 * 5 * 25 + 1

    def get_init_board(self):
        # return initial board (numpy board)
        start_board = np.zeros((7, 7, 7), dtype=np.bool)
        return start_board

    def get_board_size(self):
        # (a,b) tuple
        return 7, 7, 7

    def get_action_size(self):
        # return number of actions
        return self.actionSize

    def get_next_state(self, board: np.ndarray, player: int, action: int) -> (np.ndarray, int):
        assert board.shape == (7, 7, 7), f'An incorrect shape was given, expected was 7,7,7 given was {board.shape}'
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == (self.actionSize - 1):
            return board, -player
        flat_board = compress_tensor(board)
        b = Board(flat_board)
        b.decode_action(player, action)
        board = decompress_tensor(board, b.matrix_board)
        return board, -player

    def get_valid_moves(self, board: np.ndarray, player: int) -> list:
        # return a fixed size binary vector
        assert board.shape == (7, 7, 7), f'An incorrect shape was given, expected was 7,7,7 given was {board.shape}'
        flat_board = compress_tensor(board)
        b = Board(flat_board)
        b.board = [b.matrix_board[b.board_map[i]] for i in range(24)]
        legal_moves = b.get_legal_moves(player)
        legal_moves = list(legal_moves.reshape(-1))
        legal_moves.extend([0 if np.sum(legal_moves) > 0 else 1])
        return legal_moves

    def get_game_ended(self, board: np.ndarray, player: int):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        assert board.shape == (7, 7, 7), f'An incorrect shape was given, expected was 7,7,7 given was {board.shape}'
        flat_board = compress_tensor(board)
        b = Board(flat_board)
        b.board = [b.matrix_board[b.board_map[i]] for i in range(24)]
        if b.is_win(-player):
            return -player
        if b.is_win(player):
            return player
        valid_moves = self.get_valid_moves(board, player)
        if np.sum(valid_moves):  # game continues
            return 0
        # draw has a very little value
        return 1e-4 * player

    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        # return state if player==1, else return -state if player==-1
        assert board.shape == (7, 7, 7), f'An incorrect shape was given, expected was 7,7,7 given was {board.shape}'

        return board if player is 1 else flip_tensor(board)

    def get_symmetries(self, board: np.ndarray, pi: float) -> list:

        lala = [(board, pi)]
        return lala

    def string_representation(self, board: np.ndarray):
        assert board.shape == (7, 7, 7), f'An incorrect shape was given, expected was 7,7,7 given was {board.shape}'
        return board.tobytes()

    @staticmethod
    def get_flat_board_obj(board: np.ndarray) -> Board:
        assert board.shape == (7, 7, 7), f'An incorrect shape was given, expected was 7,7,7 given was {board.shape}'
        return Board(compress_tensor(board))
