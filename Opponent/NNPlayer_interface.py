
import numpy as np
from main_play import choose, init

def piece_translator(c):
    if c == 1:
        return 'M'
    if c == -1:
        return 'E'
    elif c == 0:
        return 'O'
    else:
        print(f"illegal piece detected {c}")


x = list('abcdefg')
y = list('7654321')


def translate_xy(s):
    our_x = x.index(s[0])
    our_y = y.index(s[1])
    return (our_x, our_y)


from Nine_Men_Morris_Alpha_2.Game import NMMLogic, NMMGame


def state_translator(board: np.array):
    """
    # return their_state representation
    :param board:
    :return:
    """
    board_obj = NMMLogic.Board()

    step_count = board_obj.decode_step_count(board)
    clean_board = board_obj.get_clean_board(board)
    my_piece_count = len(np.where(clean_board == 1)[0])
    enemy_piece_count = len(np.where(clean_board == -1)[0])
    # TODO make sure that first step is step 0
    if step_count >= 18:
        remaining = '00'
    else:
        # removed_count = my_piece_count + enemy_piece_count - step_count  # Not relevant, saved for later usage
        # mine_removed = (mine_expected - my_piece_count)
        # if step_count odd, mine player is supposed to have 1 less piece
        is_odd = (step_count % 2) == 1
        mine_expected = int((step_count - int(is_odd)) / 2)
        remaining = f'{8 - mine_expected}' + f'{int(8 - np.floor(step_count / 2))}'

    their_board = ['O'] * 24 + [remaining] + [str(my_piece_count) + str(enemy_piece_count)]
    for i in range(24):
        their_board[i] = piece_translator(clean_board[board_obj.board_map[i]])

    return ''.join(their_board)


def action_translator(is_stage2, FROMc, TOc, REMOVEc):
    # TODO: read following notes:
    # if phase 1, the FROMc is TOc
    # if phase 2, same as written
    board_obj = NMMLogic.Board()
    a = (23, 4, 24)  # indices are -1 subtracted to match max index
    if not is_stage2:
        to_xy = translate_xy(FROMc)  # variable name is not mistake
        a[0] = board_obj.board_map(to_xy)
        if REMOVEc:
            remove_xy = translate_xy(REMOVEc)
            a[2] = board_obj.board_map(remove_xy)
    else:

        from_ = board_obj.board_map(translate_xy(FROMc))
        to_ = board_obj.board_map(translate_xy(TOc))
        try:
            a[1] = board_obj.adjacent[from_].index(to_)
        except:
            print("illegal")
            return None
        if REMOVEc:
            remove_xy = translate_xy(REMOVEc)
            a[2] = board_obj.board_map(remove_xy)
    action = np.ravel_multi_index(a, dims=(24, 5, 25), order='F')

    # #TODO if step is illegal choose random
    return action

def player_wrapper():
    def init_player():
        name = 'TEST-rawest-TFR'
        print("Loading networks")
        TOnet, FROMnet, REMOVEnet, data_format = init(name)
        print("\tNetworks loaded!")
        return lambda state: (NMMLogic.Board(state).is_stage2(), choose(TOnet, FROMnet, REMOVEnet,
                                                                        state, data_format))  # (phase, action)
    ai_player = init_player()

    return lambda our_state: action_translator(ai_player(state_translator(our_state)))


if __name__ == '__main__':
    # OUR MAIN
    player = player_wrapper()
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
    s = state_translator(our_state)

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
