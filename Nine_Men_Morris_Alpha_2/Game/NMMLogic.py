'''

'''
# from bkcharts.attributes import color
import numpy as np

from Nine_Men_Morris_Alpha_2.Game.base_board import Base_mill


class Board(Base_mill):

    # list of all 8 directions on the board, as (x,y) offsets
    def __init__(self):
        super(Board, self).__init__()
        self.move_count = 0

    def is_stage_2(self):
        return self.move_count >= 18

    def is_mill(self, player: int, place: int) -> bool:
        ML = lambda player, board, pos1, pos2: board[pos1] == player and board[pos2] == player
        is_mill_or = lambda player, board, cm: ML(player, board, cm[0], cm[1]) or ML(player, board, cm[2], cm[3])
        return is_mill_or(player, self.board, self.complete_mill[place])

    def execute_move(self, player: int, remove: int, set_place: int, remove_opponent: int):
        if remove != -1:
            self.board[remove] = 0
        self.board[set_place] = player
        if remove_opponent != -1:
            self.board[remove_opponent] = 0
        self.move_count += 1


    def get_legal_moves(self, player: int) -> np.ndarray:
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """


        return

    def make_a_move(self, player: int, move: np.ndarray):
        pass

    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """


        pass

