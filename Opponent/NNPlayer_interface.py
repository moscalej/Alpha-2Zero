import numpy as np
from main_play import choose, init
from dataprocessing import *
from Nine_Men_Morris_Alpha_2.Game import NMMLogic, NMMGame


def piece_translator(c):
    if c == 1:
        return 'M'
    if c == -1:
        return 'E'
    elif c == 0:
        return 'O'
    else:
        print(f"illegal piece detected {c}")


#
# x = list('abcdefg')
# y = list('7654321')
#
#
# def translate_xy(s):
#     our_x = x.index(s[0])
#     our_y = y.index(s[1])
#     return (our_x, our_y)

def translate_(i):
    return i - 1


def state_translator(board: np.array):
    """
    # return their_state representation
    :param board:
    :return:
    """
    init_pieces_n = 9
    board_obj = NMMLogic.Board()

    encoded_step_count = board_obj.decode_step_count(board)
    clean_board = board_obj.get_clean_board(board)
    my_piece_count = len(np.where(clean_board == 1)[0])
    enemy_piece_count = len(np.where(clean_board == -1)[0])
    if my_piece_count + enemy_piece_count >= 5:
        step_count = encoded_step_count + board_obj.count_offset
    else:
        step_count = my_piece_count + enemy_piece_count
    # TODO make sure that first step is step 0
    if step_count >= 18:
        remaining = '00'
    else:
        # if step_count odd, mine player is supposed to have 1 less piece
        mine_expected = int((step_count - (step_count % 2)) / 2)
        enemy_expected = np.ceil(step_count / 2)
        remaining = f'{init_pieces_n - mine_expected}' + f'{int(init_pieces_n - enemy_expected)}'

    their_board = ['O'] * 24 + [remaining] + [str(my_piece_count) + str(enemy_piece_count)]
    for i in range(24):
        their_board[i] = piece_translator(clean_board[board_obj.board_map_alt[i]])

    return ''.join(their_board)


def action_translator(is_stage2, TOc, FROMc, REMOVEc):
    # TODO: read following notes:
    # if phase 1, the FROMc is TOc
    # if phase 2, same as written
    board_obj = NMMLogic.Board()
    get_a = lambda x: board_obj.inv_map[board_obj.board_map_alt[translate_(x)]]  # get location in our board array
    a = [23, 4, 24]  # indices are -1 subtracted to match max index
    if not is_stage2:
        # set piece
        a[0] = get_a(FROMc)  # variable name is not mistake
        if REMOVEc:
            a[2] = get_a(REMOVEc)
    else:
        # move piece
        from_ = get_a(FROMc)
        to_ = get_a(TOc)
        try:
            a[1] = board_obj.adjacent[from_].index(to_)
        except:
            print("illegal")
        if REMOVEc:
            a[2] = get_a(REMOVEc)
    action = np.ravel_multi_index(a, dims=(24, 5, 25), order='F')

    return action


def NN_player_wrapper(name='TEST-rawest-TFR'):
    print("Loading networks")
    TOnet, FROMnet, REMOVEnet, data_format = init(name)
    print("\tNetworks loaded!")

    their_player = lambda state: choose(TOnet, FROMnet, REMOVEnet,
                                        state, data_format)  # (phase, action)

    def opp_player(our_state, data_format=data_format):
        g = NMMGame.MenMorris(9)
        b = g.get_board_obj(our_state)
        # the step count would be wrong for the initial 4 steps, but it doesn't effect the stage
        step_count = b.decode_step_count(b.get_clean_board(our_state)) + b.count_offset
        isStage2 = step_count >= 18
        phase = 2 if isStage2 else 1
        # try:
        their_state = process_game_line(state_translator(our_state))
        TO, FROM, REMOVE = their_player(their_state)
        return action_translator(isStage2, FROM, TO, REMOVE)
        # except Exception as e:
        #     print(f"opponent failed with: {e}")
        #     pi_mask = np.array(g.getValidMoves(our_state, 1)) * 1
        #     pi = pi_mask / np.sum(pi_mask)  # uniform distribution of all valid moves
        #     return np.random.choice(3001, p=pi)

    return opp_player


def board_format_converter_TO(s):
    state_obj = process_states_dataset_line(s)
    their_board = state_obj.to_board()
    their_board = their_board.replace('-', '0')
    their_board = their_board.replace(' ', '0')
    their_board = their_board.replace('|', '0')
    their_board = their_board.replace('M', '1')
    their_board = their_board.replace('E', '2')
    their_board = their_board.replace('O', '0')
    their_clean_board = their_board.split('\n')[:-1]
    their_clean_board = [np.array([int(piece) if int(piece) != 2 else -1 for piece in list(row)]) for row in
                         their_clean_board]
    result = np.array(their_clean_board)
    return result

    # def init_player():
    #     name = 'TEST-rawest-TFR'
    #     print("Loading networks")
    #     TOnet, FROMnet, REMOVEnet, data_format = init(name)
    #     print("\tNetworks loaded!")
    #     return lambda state: choose(TOnet, FROMnet, REMOVEnet,
    #                                 state, data_format)

    #
    # process_game_line()
    # ai_player = init_player()
