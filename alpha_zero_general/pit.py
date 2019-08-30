"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
from MCTS import MCTS
from Arena import Arena

from Nine_Men_Morris_Alpha_2.Game.NMMGame import MenMorris as Game
from Nine_Men_Morris_Alpha_2.keras.NNet import NNetWrapper as NeuralNetwork
from Opponent.NNPlayer_interface import NN_player_wrapper
import numpy as np

from utils import dotdict
from Nine_Men_Morris_Alpha_2.Game.NMMLogic import Board

args = dotdict({
    'numIters': 1000,
    'numEps': 400,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 0.4,
    'epochs': 40,
    'checkpoint': r'C:\Users\amoscoso\Documents\Technion\deeplearning\Alpha-2Zero\alpha_zero_general\temp',
    'load_model': False,
    'load_folder_file': ('.\\temp', 'temp.pth.tar'),
    'load_folder_Sample': ('.\\temp', 'checkpoint_64.pth.tar.tar'),
    'numItersForTrainExamplesHistory': 20,
})

game = Game(men_count=9)

neural_network = NeuralNetwork(game)
neural_network.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')

our_player = MCTS(game, neural_network, args)
other_player = NN_player_wrapper()


def player_ramdon(board):
    game_2 = Game(men_count=9)
    valids = game_2.get_valid_moves(board,1)
    valids_ind = np.where(valids)
    choise = np.random.choice(valids_ind[0])
    return  choise


print('Let the fight Begin')
arena = Arena(lambda x: np.argmax(our_player.get_action_prob(x)),
              player_ramdon, game, lambda x: Board(x).verbose_game(x),"Alpha_Zero", "Ramdom")

print(arena.playGames(2, verbose=True))

print('Let the fight Begin')
arena = Arena(other_player,
              player_ramdon, game, lambda x: Board(x).verbose_game(x),"Other Network", "Ramdom")

print(arena.playGames(2, verbose=True))
