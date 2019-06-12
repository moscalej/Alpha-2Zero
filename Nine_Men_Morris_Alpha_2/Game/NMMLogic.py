# from bkcharts.attributes import color
import numpy as np

from Nine_Men_Morris_Alpha_2.Game.base_board import Base_mill
from termcolor import colored


def int_to_bin_string(i):
    if i == 0:
        return "0"
    s = ''
    while i:
        if i & 1 == 1:
            s = "1" + s
        else:
            s = "0" + s
        i //= 2
    return s


class Board(Base_mill):

    # list of all 8 directions on the board, as (x,y) offsets
    def __init__(self, matrix_b=None):
        super(Board, self).__init__()
        if matrix_b is not None:
            self.matrix_board = matrix_b

    def is_mill(self, player: int, place: int, board_: list) -> bool:
        ML = lambda player, board, pos1, pos2: board[pos1] == player and board[pos2] == player
        is_mill_or = lambda player, board, cm: ML(player, board, cm[0], cm[1]) or ML(player, board, cm[2], cm[3])
        return is_mill_or(player, board_, self.complete_mill[place])

    def execute_move(self, player: int, remove: int, set_place: int, remove_opponent: int):
        if remove != 24:  # moving a piece on the board
            self.matrix_board[self.board_map[remove]] = 0
            self.board[remove] = 0

        self.matrix_board[self.board_map[set_place]] = player  # setting a piece on the board
        self.board[set_place] = player

        if remove_opponent != 24:
            self.matrix_board[self.board_map[remove_opponent]] = 0  # removing opponent piece
            self.board[remove_opponent] = 0
        self.encode_next()

    def get_legal_moves(self, player: int) -> np.ndarray:
        """
        create mask of legal moves
        :param player:
        :type player:
        :return:
        :rtype:
        """
        stage2 = self.is_stage2()
        action_mask = np.zeros((24, 5, 25), dtype=bool)
        # if stage 1 add set options
        array_board = np.array(self.board)
        if not stage2:
            legal_pos = np.where(array_board == 0)[0]
            for pos in legal_pos:
                if self.is_mill(player, pos, self.board):  # current selection completes a mill
                    opp_pos = np.where(self.board == -player)[0]
                    opp_pos = [opp_p for opp_p in opp_pos if
                               not self.is_mill(-player, opp_p, self.board)]  # can't remove opponent in mill
                    action_mask[pos, -1, opp_pos] = True
                else:
                    action_mask[pos, -1, -1] = True  # place piece on board
        else:
            from_pos_cands = np.where(array_board == player)[0]
            for from_pos in from_pos_cands:
                mill_cands = [(orient, adj) for orient, adj in enumerate(self.adjacent[from_pos]) if
                              adj is not None and self.board[adj] == 0]  # TODO added not, need to validate
                if_played_board = self.board.copy()
                if_played_board[from_pos] = 0
                for (orient, adj) in mill_cands:
                    if self.is_mill(player, adj, if_played_board):
                        opp_pos = np.where(array_board == -player)[0]
                        opp_pos = [opp_p for opp_p in opp_pos if not self.is_mill(-player, opp_p, if_played_board)]
                        action_mask[from_pos, orient, opp_pos] = True
                    else:
                        action_mask[from_pos, orient, -1] = True

        return action_mask

    def decode_action(self, player: int, action_code: int):
        piece, action, remove = np.unravel_index(action_code, (24, 5, 25))  # there is additional code for end of game
        if action == 4:  #
            self.execute_move(player, remove=24, set_place=piece, remove_opponent=remove)
        else:
            self.execute_move(player, remove=piece, set_place=self.adjacent[piece][action], remove_opponent=remove)

    def is_win(self, player):
        """  TODO: make sure that this is used in the end of player turn, and evaluated with the new board for the player that played
        Game ends only in stage 2, in one of two cases:
            1) the opponent has 2 pieces
            2) the opponent has no legal moves left
        :param player: The player that just played
        :return: bool indicating if player has won
        """
        if not self.is_stage2():
            return False
        board = self.get_clean_board(self.matrix_board)
        unique, counts = np.unique(board, return_counts=True)
        opp_count = dict(zip(unique, counts))[-player]
        if opp_count <= 2 or not np.sum(self.get_legal_moves(-player)):
            print(f"player {player} wins")
            return True

        return False

    def encode_next(self):
        """
        updates board to have count + 1
        count is shifted  steps 0,1,2,3 => 0, and then 4=>1, 5=>2, ect..
        :return: None
        """
        steps = self.decode_step_count()
        if steps >= 18 - 3:
            return
        if steps == 0:
            debug = np.sum(np.sum(np.abs(self.matrix_board)))
            if debug < 4:
                return
        self.encode_step_count(steps + 1)

    def encode_step_count(self, steps):
        bin_str = int_to_bin_string(steps)
        for ind, val in enumerate(bin_str[::-1]):
            self.matrix_board[self.bits_map[ind]] = val

    def decode_step_count(self):
        """
        decode steps count from board encoding
        count is shifted  steps 0,1,2,3 => 0, and then 4=>1, 5=>2, ect..
        :return: int number of steps
        """
        # TODO decide which one is better.. not crucial
        # steps = 0
        # for key_pow, val_coor  in self.read_bits.items():
        #     steps += (self.matrix_board[val_coor] * 2) ** key_pow
        # return steps

        bit3 = self.matrix_board[self.read_bits[3]]
        bit2 = self.matrix_board[self.read_bits[2]]
        bit1 = self.matrix_board[self.read_bits[1]]
        bit0 = self.matrix_board[self.read_bits[0]]
        return int(f'0b{bit3}{bit2}{bit1}{bit0}', 2)

    def is_stage2(self):
        """
        check if
        :return: bool answering "is stage 2?"
        """
        return self.decode_step_count() == 18 - 3  # stage 2 from 18th step count is shifted

    def cononical_board(self, player):
        step = self.decode_step_count()
        self.matrix_board *= -1  # TODO: originaly it was *= player, validate
        self.encode_step_count(step)
        return self.matrix_board

    def get_clean_board(self, board_):
        board = np.copy(board_)
        for _, bit_coor in self.bits_map.items():
            board[bit_coor] = 0
        return board

    def verbal_action_decode(self, action_code):
        if action_code == 24 * 5 * 25:
            print(f"game ended action code: {action_code}")
            return
        action_types = ['up', 'down', 'left', 'right']
        piece, action, remove = np.unravel_index(action_code, (24, 5, 25))
        describe_moves = []
        if action == 4:
            describe_moves.append(f"Set a piece in {self.board_map[piece]}")
        else:
            describe_moves.append(f"Move from {self.board_map[piece]} {action_types[action]}"
                                  f" to {self.board_map[self.adjacent[piece][action]]}")

        if remove != 24:
            describe_moves.append(f"and Remove from {self.board_map[remove]}")

    def print_board(self, board, action_code=None):
        withAction = action_code is not None
        set = []
        removed = []
        legal_inds = list(self.board_map.values())
        if withAction:
            if action_code == 24 * 5 * 25:
                print(f"game ended action code: {action_code}")
                return
            piece, action, remove = np.unravel_index(action_code, (24, 5, 25))
            action_types = ['up', 'down', 'left', 'right']

            describe_moves = []

            if action == 4:
                describe_moves.append(f"Set a piece in {self.board_map[piece]}")
                set.append(self.board_map[piece])

            else:
                removed.append(self.board_map[piece])
                set.append(self.board_map[self.adjacent[piece][action]])
                describe_moves.append(f"Move from {self.board_map[piece]} {action_types[action]}"
                                      f" to {self.board_map[self.adjacent[piece][action]]}")

            if remove != 24:
                removed.append(self.board_map[remove])
                describe_moves.append(f"and Remove from {self.board_map[remove]}")
            print('\n'.join(describe_moves))
            b = Board(np.copy(board))
            b.decode_action(1, action_code)
            next_board = b.matrix_board
            print('legend:', colored('removed', 'red'), colored('Placed', 'yellow'), colored('Player 1', 'blue'),
                  colored('Player 2', 'magenta'))
        else:
            next_board = board

        def getColor(x_, y_, player):
            if (x_, y_) not in legal_inds:
                return 'blue', ' '
            if player == 1:
                s = 'X'
                color = 'blue'
            elif player == -1:
                s = 'O'
                color = 'magenta'
            else:
                color = 'white'
                s = '-'
            if (x_, y_) in set:
                s = 'X'
                color = 'yellow'
            if (x_, y) in removed:
                s = '-'
                color = 'red'
            return color, s



        next_board = self.get_clean_board(next_board)
        n = next_board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = next_board[y][x]  # get the piece to print
                color, s = getColor(x, y, piece)
                if piece == -1:
                    print(colored(f"{s} ", color), end="")
                elif piece == 1:
                    print(colored(f"{s} ", color), end="")
                else:
                    if x == n:
                        print(colored(f"{s}", color), end="")
                    else:
                        print(colored(f"{s} ", color), end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
