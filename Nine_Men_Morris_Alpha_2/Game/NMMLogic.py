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
        if remove != 24:
            self.board[remove] = 0
        self.board[set_place] = player
        if remove_opponent != 24:
            self.board[remove_opponent] = 0
        self.move_count += 1

    def get_legal_moves(self, player: int, stage2: bool) -> np.ndarray:
        """
        create mask of legal moves
        :param player:
        :type player:
        :return:
        :rtype:
        """

        action_mask = np.zeros((24, 5, 25), dtype=bool)
        # if stage 1 add set options
        if not stage2:
            legal_pos = np.where(self.board == 0)[0]
            for pos in legal_pos:
                if self.is_mill(player, pos):
                    opp_pos = np.where(self.board == -player)[0]
                    opp_pos = [opp_p for opp_p in opp_pos if not self.is_mill(-player, opp_p)]
                    action_mask[pos, -1, opp_pos] = True
                else:
                    action_mask[pos, -1, -1] = True
        else:
            from_pos_cands = np.where(self.board == player)[0]
            for from_pos in from_pos_cands:
                mill_cands = [(orient, adj) for orient, adj in enumerate(self.adjacent[from_pos]) if
                              adj != None and self.board[adj] == 0]
                if_played_board = self.board.copy()
                if_played_board[from_pos] = 0
                for (orient, adj) in mill_cands:
                    if self.is_mill(player, adj):
                        opp_pos = np.where(self.board == -player)[0]
                        opp_pos = [opp_p for opp_p in opp_pos if not self.is_mill(-player, opp_p)]
                        action_mask[from_pos, orient, opp_pos] = True
                    else:
                        action_mask[from_pos, orient, -1] = True

        return action_mask

    def decode_action(self, player: int, action_code: int):
        # assert action.shape == (24, 5, 25)  # TODO action is int
        piece, action, remove = np.unravel_index(action_code, (24, 5, 25))
        if action is 4:
            self.execute_move(player, 24, piece, remove)
        else:
            self.execute_move(player, piece, self.adjacent[action], remove)

    def is_win(self, player, stage2):

        # if others player number of pieces is 2 you win

        unique, counts = np.unique(self.matrix_board, return_counts=True)
        opp_count = dict(zip(unique, counts))[-player]
        if opp_count < 2:
            return True
        if not np.sum(self.get_legal_moves(player, stage2)):
            return True
        return False

