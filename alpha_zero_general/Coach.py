from collections import deque
from Arena import Arena
from MCTS import MCTS
from tqdm import tqdm
import numpy as np
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import sys
from Nine_Men_Morris_Alpha_2.Game.NMMLogic import Board

PATH = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Alpha-2Zero\alpha_zero_general\data'


def winner(outcome: int, player: int, current: int) -> int:
    """
    Returns the winner
    :param outcome: outcome = 1 player == current --> -1
    :param player:
    :param current:
    :type current:
    :return:
    :rtype:
    """
    if outcome == 1:
        if player == 1:
            return 1
        else:
            return -1
    else:
        if player == -1:
            return 1
        else:
            return -1


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples
        # from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.curPlayer = None
        self.name = "Jorge"

    def execute_episode(self, verbose) -> list:
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonical_board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        board = self.game.get_init_board()
        self.curPlayer = 1
        episode_step = 0
        verbose = True
        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, self.curPlayer)
            temp = int(episode_step < self.args.tempThreshold)
            pi, counts , values = self.mcts.get_action_prob(canonical_board.copy(), temp=temp)
            train_examples.append((canonical_board.copy(), self.curPlayer, pi.copy(), None))
            action = np.random.choice(len(pi), p=pi)

            #########################  TODO: remove after testing
            # # action = np.argmax(pi)
            # moves_verbose.append(b_obj.verbose_game(canonical_board, action, no_board=True))
            # if episode_step % 5 == 0 and episode_step != 1 and sample_collection is True:
            #     timestamp = datetime.datetime.now().strftime("%H_%M_%S")
            #     np.save(f'{PATH}\\{timestamp}.npy', canonical_board)
            #     original = sys.stdout
            #     sys.stdout = open(f"{PATH}\\{timestamp}.txt", 'w')
            #     # print(f"step count {b.decode_step_count()}:")
            #     print('\n'.join(moves_verbose))
            #     b_obj.verbose_game(canonical_board)
            #     sys.stdout.close()
            #     sys.stdout = original
            # ########################

            if verbose:
                self.game.print_board(canonical_board, action)
                self.game.log_moves(pi, values,counts)
                print("we choose ", action)

            new_board, new_player = self.game.get_next_state(board, self.curPlayer, action)
            if verbose:
                self.game.print_board(new_board)

            response = self.game.get_game_ended(new_board, 1)

            if response != 0:
                last_player = new_player
                print("Simulated game end")
                print(f"Number of iterations was {episode_step}")
                print(f"Last move was done by {self.curPlayer}")
                self.game.print_board(new_board)
                winer = {
                    (1, 1): 'player 1',
                    (1, -1): 'player -1',
                    (-1, 1): 'player 1',
                    (-1, -1): 'player -1',
                }[last_player, response]
                print(f"This move make the winner to be{winer}")
                vals = []
                for board_m, player_, pi_, _ in train_examples:
                    val = (board_m, pi_, winner(response, player_, new_player))
                    vals.append(val)
                print(vals[-1][2])

                return vals
            self.curPlayer = new_player
            board = new_board

    def learn(self, verbose):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

                for eps in tqdm(range(self.args.numEps)):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    results = self.execute_episode(verbose)
                    if results is not None:
                        iteration_train_examples += results

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iteration_train_examples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.trainExamplesHistory:
                train_examples.extend(e)
            shuffle(train_examples)

            # # training new network, keeping a copy of the old one
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp1.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp1.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            # nmcts = MCTS(self.game, self.nnet, self.args)

            # print('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0, )),
            #               lambda x: np.argmax(nmcts.get_action_prob(x, temp=0, )), self.game)
            # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            #
            # print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # # if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:

            if False:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return f'checkpoint_{iteration + 1}.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples1")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples1"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
