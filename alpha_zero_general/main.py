from Coach import Coach

# from othello.OthelloGame import OthelloGame as Game
from Nine_Men_Morris_Alpha_2.Game.NMMGame import MenMorris as Game
from Nine_Men_Morris_Alpha_2.keras.NNet import NNetWrapper as nn
from utils import *

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
    'load_folder_file': ('.\\temp', 'checkpoint_35.pth.tar'),
    'load_folder_Sample': ('.\\temp', 'checkpoint_35.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    # debug
    import sys
    # print(sys.getrecursionlimit())
    # sys.setrecursionlimit(10000)
    #
    g = Game(men_count=9)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
