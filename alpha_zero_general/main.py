from alpha_zero_general.Coach import Coach

# from othello.OthelloGame import OthelloGame as Game
from Nine_Men_Morris_Alpha_2.Game.NMMGame import MenMorris as Game
from Nine_Men_Morris_Alpha_2.keras.NNet import NNetWrapper as NeuralNetwork
from alpha_zero_general.utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 300000,
    'numMCTSSims': 25,
    'arenaCompare': 10,
    'cpuct': 0.01,
    'n_importance' : 0.1,
    'epochs': 60,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('.\\temp', 'checkpoint_1.pth.tar'),
    'load_folder_Sample': ('.\\temp1', 'checkpoint_5.pth.tar.examples'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    # debug
    import sys
    # print(sys.getrecursionlimit())
    # sys.setrecursionlimit(10000)
    #
    game = Game(men_count=9)
    neural_network = NeuralNetwork(game)

    if args.load_model:
        neural_network.load_checkpoint(args.load_folder_file[0], 'best.pth.tar')

    coach = Coach(game, neural_network, args)
    if args.load_model:
        # coach.loadTrainExamples()

        print("Load trainExamples from file")
        # coach.loadTrainExamples()
    coach.learn(verbose = False)
