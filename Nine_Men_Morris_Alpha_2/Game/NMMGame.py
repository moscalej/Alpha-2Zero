import numpy as np
from alpha_zero_general.Game import Game
from Nine_Men_Morris_Alpha_2.Game.NMMLogic import Board


def compress_tensor(tesor_board: np.ndarray) -> np.ndarray:  # TODO Alejandro
    """
    Compress state representation to 7 by 7 board
    :param tesor_board:
    :return: 7 by 7 np.ndarray
    """
    assert tesor_board.shape == (7, 7, 7), "Invalid Input Shape "
    pass

def decompress_tensor(prev_tensor_board: np.ndarray, new_board: np.ndarray) -> np.ndarray: # TODO Alex
    """
    Compress state representation to 7 by 7 by 7 tensor board
    :param prev_tensor_board:
    :param new_board:
    :return: 7 by 7 by 7 np.ndarray
    """
    assert new_board.shape == (7, 7), "Invalid Input Shape "
    assert prev_tensor_board.shape == (7, 7, 7), "Invalid Input Shape "

    pass

def get_canonical_tensor(tensor_board: np.ndarray) -> np.ndarray:
    """
    Compress state representation to 7 by 7 by 7 tensor board
    :param prev_tensor_board:
    :param new_board:
    :return: 7 by 7 by 7 np.ndarray
    """
    assert tensor_board.shape == (7, 7, 7), "Invalid Input Shape "

    pass



class MenMorris(Game):
    def __init__(self, men_count):
        super(MenMorris, self).__init__()
        self.men_count = men_count
        self.actionSize = 24 * 5 * 25 + 1


    def get_init_board(self):
        # return initial board (numpy board)
        b = Board()
        return b.matrix_board  # TODO: wrap with decompress

    def get_board_size(self):
        # (a,b) tuple
        return 7, 7  # TODO: consider changing

    def get_action_size(self):
        # return number of actions
        return self.actionSize

    def get_next_state(self, board: np.ndarray, player: int, action: int) -> (np.ndarray, int):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # TODO: wrap with compress
        if action == (self.actionSize - 1):
            return board, -player
        b = Board(board.copy())
        b.decode_action(player, action)
        # board = np.copy(b.matrix_board)
        return b.matrix_board, -player  # TODO wrap with decompress

    def get_valid_moves(self, board: np.ndarray, player: int) -> list:
        # return a fixed size binary vector
        # TODO: wrap with compress
        b = Board(board.copy())
        b.board = [b.matrix_board[b.board_map[i]] for i in range(24)]
        legal_moves = b.get_legal_moves(player)
        legal_moves = list(legal_moves.reshape(-1))
        legal_moves.extend([0 if np.sum(legal_moves) > 0 else 1])
        return legal_moves

    def get_game_ended(self, board: np.ndarray, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        # TODO: wrap with compress
        b = Board(board.copy())
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

    # TODO: wrap with get_canonical_tensor
    def get_canonical_form(self, board: np.ndarray, player):
        # return state if player==1, else return -state if player==-1
        b = Board(board.copy())
        return b.canonical_board(player)

    def get_symmetries(self, board: np.ndarray, pi: float) -> list:
        # mirror, rotational
        # assert(len(pi) == self.n**2+1)  # 1 for pass
        # pi_board = np.reshape(pi[:-1], (24, 5, 25))
        # l = []
        # board = board.copy()
        # for i in range(1, 5):
        #     for j in [True, False]:
        #         newB = np.rot90(board, i)
        #         newPi = np.rot90(pi_board, i)  # TODO check if this rotation is correct
        #         if j:
        #             newB = np.fliplr(newB)
        #             newPi = np.fliplr(newPi)  # TODO check if this flip is correct
        #         l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        l = [(board, pi)]
        return l

    def string_representation(self, board: np.ndarray):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def get_board_obj(self, board):
        # TODO: wrap with compress
        return Board(board)
