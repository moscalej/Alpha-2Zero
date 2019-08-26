
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
from MCTS import MCTS
from Arena import Arena
from Nine_Men_Morris_Alpha_2.Game.NMMGame import MenMorris as Game
from Nine_Men_Morris_Alpha_2.keras.NNet import NNetWrapper as NeuralNetwork
import numpy as np
args = dotdict({
    'numIters': 1000,
    'numEps': 400,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 24,
    'arenaCompare': 40,
    'cpuct': 10,
    'epochs': 40,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('.\\temp', 'checkpoint_65.pth.tar'),
    'load_folder_Sample': ('.\\temp', 'checkpoint_65.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


game = Game(men_count=9)
neural_network = NeuralNetwork(game)
neural_network.nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
nmcts = MCTS(game, neural_network.nnet, args)


print('PITTING AGAINST PREVIOUS VERSION')
arena = Arena(lambda x: np.argmax(nmcts.get_action_prob(x, temp=0, )),
              lambda x: np.argmax(nmcts.get_action_prob(x, temp=0, )), game)



print(arena.playGames(2, verbose=True))
