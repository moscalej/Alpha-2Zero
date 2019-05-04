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
        return

    def getNextState(self, board: np.ndarray, player: int, action: int) -> (np.ndarray, int):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.actionSize - 1:  #?? TODO validate probably end of game
            return (board, -player)

        b = NMMLogic.Board()
        b.matrix_board = np.copy(board)
        move = b.decode_action(player, action)
        b.execute_move(move, player)
        return (b.matrix_board, -player)

    def getValidMoves(self, board, player):
        # TODO description here might be insufficient, compare to previous game implementations
        # return a fixed size binary vector
        b = NMMLogic.Board()
        b.matrix_board = np.copy(board)  #TODO ask coach for game stage??
        legalMoves = b.get_legal_moves(player)
        legalMoves = legalMoves.reshape(-1)
        # TODO extend action array, if no legalMoves add last element as 1, otherwise 0?
        return legalMoves

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = NMMLogic.Board()
        b.matrix_board = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
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
