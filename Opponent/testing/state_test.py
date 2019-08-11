import numpy as np
from NNPlayer_interface import *

samples = [20]
for sample in samples:
    base = fr'C:/Users/afinkels/Desktop/private/Technion/Master studies/Project in Deep Learning/Alpha-2Zero-master/Alpha-2Zero/Opponent/testing/our_board_samples/{sample}'
    path_txt = fr'{base}.txt'
    path_npy = fr'{base}.npy'
    # print recorded board
    with open(path_txt, 'r') as fin:
        print(fin.read())
    # load recorded board (with step encoding)
    board = np.load(path_npy)
    s = state_translator(board)
