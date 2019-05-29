from __future__ import print_function

import numpy as np
from alpha_zero_general.Game import Game
from Nine_Men_Morris_Alpha_2.Game import NMMLogic

class MenMorris(Game):
    def __init__(self, men_count):
        self.men_count = men_count
        self.actionSize = 24 * 5 * 25 + 1

    def getInitBoard(self):
        # return initial board (numpy board)
        b = NMMLogic.Board()
        return b.matrix_board

    def getBoardSize(self):
        # (a,b) tuple
        return (7, 7)

    def getActionSize(self):
        # return number of actions
        return self.actionSize

    def getNextState(self, board: np.ndarray, player: int, action: int) -> (np.ndarray, int):
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        if action == (self.actionSize - 1):  #?? TODO validate probably end of game
            return (board, -player)
        b = NMMLogic.Board()
        b.matrix_board = np.copy(board)
        b.decode_action(player, action)
        # board = np.copy(b.matrix_board)
        return (b.matrix_board, -player)

    def getValidMoves(self, board, player, stage2):
        # return a fixed size binary vector
        b = NMMLogic.Board()
        b.matrix_board = np.copy(board)
        b.board = [b.matrix_board[b.board_map[i]] for i in range(max(list(b.board_map.keys()))+1)]
        legalMoves = b.get_legal_moves(player, stage2)
        legalMoves = list(legalMoves.reshape(-1))
        legalMoves.extend([0 if np.sum(legalMoves) > 0 else 1])
        return legalMoves

    def getGameEnded(self, board, player, stage2):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = NMMLogic.Board()
        b.matrix_board = np.copy(board)

        if b.is_win(player, stage2):
            return player
        if b.is_win(-player, stage2):
            return -player
        valid_moves = self.getValidMoves(board, player, stage2)
        if np.sum(valid_moves):  # game continues
            return 0
        # draw has a very little value
        return 1e-4 * player

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        # assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (24, 5, 25))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)  # TODO check if this rotation is correct
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi) # TODO check if this flip is correct
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

def display(board):
    n = board.shape[0]
    print("   ", end="")
    for y in range(n):
        print (y,"", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print ("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("X ",end="")
            elif piece == 1: print("O ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print ("-", end="-")
    print("--")
