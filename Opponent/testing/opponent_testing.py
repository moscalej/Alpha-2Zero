import numpy as np
from NNPlayer_interface import *
import os
from dataprocessing import *

mine = 'M'
enemy = 'E'
empty = 'O'
state_action_sep = '-'
# State Format
# first 24 characters board state,
# next two are the remaining pieces of each player (Mine, Enenmy)
# Last Two are number of checkers players have on board
our_state = np.zeros([7, 7], dtype=int)
state_example = 'OEOOEOEEMOOOOMOOEEMOOMOO4346'
action_example = 'b6b4e4'
action_example2 = 'c3'
state_action_example = state_example + state_action_sep + action_example2
state_obj = process_states_dataset_line(state_action_example)


def test_board_translations():
    base = fr'C:/Users/afinkels/Desktop/private/Technion/Master studies/Project in Deep Learning/Alpha-2Zero-master/Alpha-2Zero/Opponent/testing/our_board_samples/'
    samples = list(set([file.split('.')[0] for file in os.listdir(base)]))

    for sample in samples:
        path_txt = os.path.join(base, fr'{sample}.txt')
        path_npy = os.path.join(base, fr'{sample}.npy')
        # print recorded board
        print("Original Board")
        with open(path_txt, 'r') as fin:
            print(fin.read())
        # load recorded board (with step encoding)
        our_board = np.load(path_npy)
        b = NMMLogic.Board()
        step_count = b.decode_step_count(our_board) + b.count_offset
        our_clean = b.get_clean_board(our_board)
        s = state_translator(our_board)
        translated_board = board_format_converter_TO(s)
        print('X-Mine, O-Opponent')
        print(f"Test: Pieces correctness for Sample: {sample}: passed?: {np.array_equal(translated_board, our_clean)}")
        print(
            f"Stage:{2 if step_count >= 18 else 1} |Observed step: {step_count} | predicted remaining\ set pieces: M-{s[24]}\{s[26]} E-{s[25]}\{s[27:]} ")
        print(f"Test: state correctness for Sample: {sample}: passed?: ")


def test_action_translations():
    pass


def test_player():
    player = NN_player_wrapper()
    base = fr'C:/Users/afinkels/Desktop/private/Technion/Master studies/Project in Deep Learning/Alpha-2Zero-master/Alpha-2Zero/Opponent/testing/our_board_samples/'
    samples = list(set([file.split('.')[0] for file in os.listdir(base)]))
    # samples = ['23_47_32']
    b = NMMLogic.Board()

    for sample in samples:
        # path_txt = os.path.join(base, fr'{sample}.txt')
        print(f"Sample: {sample}")
        path_npy = os.path.join(base, fr'{sample}.npy')
        our_board = np.load(path_npy)
        action = player(our_board)
        b.verbose_game(our_board, action)

    pass


if __name__ == "__main__":
    test_player()
    test_board_translations()
