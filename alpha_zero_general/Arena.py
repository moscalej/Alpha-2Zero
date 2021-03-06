import numpy as np
import time


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None, name_player_1=None, name_player_2=None):

        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """

        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.name_1 = name_player_1
        self.name_2 = name_player_2

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        players_names = [self.name_2, None, self.name_1]

        curPlayer = 1
        board = self.game.get_init_board()
        it = 0
        continue_game = True
        while continue_game:
            it += 1
            if False:
                assert self.display
                print(f"Turn {it}, Player  {curPlayer}")
                self.display(board)
            canonical_board = self.game.get_canonical_form(board, curPlayer)
            action = players[curPlayer + 1](canonical_board)

            valids = self.game.get_valid_moves(self.game.get_canonical_form(board, curPlayer), 1)
            # self.display(board)
            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, next_player = self.game.get_next_state(board, curPlayer, action)
            # self.display(board)
            if( self.game.get_game_ended(board, 1) != 0):
                continue_game = False
                if verbose:
                    assert (self.display)
                    print("Game over: Turn ", str(it), "Result ", str(self.game.get_game_ended(board, 1)))
                    print(f'The Looser of the game is{players_names[curPlayer + 1]}')
                    self.display(board)
            curPlayer = next_player
        return self.game.get_game_ended(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1


        print(f'player {1} state--> won: {oneWon} loss: {twoWon}')
        self.player1, self.player2 = self.player2, self.player1
        self.name_1, self.name_2 = self.name_2, self.name_1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            end = time.time()

        return oneWon, twoWon, draws
