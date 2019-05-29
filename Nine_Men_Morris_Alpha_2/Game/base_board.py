import numpy as np


class Base_mill:
    def __init__(self):
        "Set up initial board configuration."

        self.matrix_board = np.zeros([7, 7], dtype=int)
        self.board_map = {
            0: (0, 0), 1: (0, 3), 2: (0, 6), 8: (1, 1),
            9: (1, 3), 10: (1, 5), 16: (2, 2), 17: (2, 3),
            18: (2, 4), 3: (3, 0), 11: (3, 1), 19: (3, 2),
            20: (3, 4), 12: (3, 5), 4: (3, 6), 21: (4, 2),
            22: (4, 3), 23: (4, 4), 13: (5, 1), 14: (5, 3),
            15: (5, 5), 5: (6, 0), 6: (6, 3), 7: (6, 6)
        }

        self.board = np.zeros(24)

        self.adjacent = {
            0: [None, 3, None, 1],
            1: [None, 9, 0, 2],
            2: [None, 4, 1, None],
            3: [0, 5, None, 11],
            4: [2, 7, 12, None],
            5: [3, None, None, 6],
            6: [14, None, 5, 7],
            7: [4, None, 6, None],
            8: [None, 11, None, 9],
            9: [1, 17, 8, 10],
            10: [None, 12, 9, None],
            11: [8, 13, 3, 19],
            12: [10, 15, 20, 4],
            13: [11, None, None, 14],
            14: [22, 6, 13, 15],
            15: [12, None, 14, None],
            16: [None, 19, None, 17],
            17: [9, None, 16, 18],
            18: [None, 20, 17, None],
            19: [16, 21, 11, None],
            20: [18, 23, None, 12],
            21: [19, None, None, 22],
            22: [None, 14, 21, 23],
            23: [20, None, 22, None]
        }
        self.complete_mill = {
            0: [1, 2, 3, 5],
            1: [0, 2, 9, 17],
            2: [0, 1, 4, 7],
            3: [0, 5, 11, 19],
            4: [2, 7, 12, 20],
            5: [0, 3, 6, 7],
            6: [5, 7, 14, 22],
            7: [2, 4, 5, 6],
            8: [9, 10, 11, 13],
            9: [8, 10, 1, 17],
            10: [8, 9, 12, 15],
            11: [3, 19, 8, 13],
            12: [20, 4, 10, 15],
            13: [8, 11, 14, 15],
            14: [13, 15, 6, 22],
            15: [13, 14, 10, 12],
            16: [17, 18, 19, 21],
            17: [1, 9, 16, 18],
            18: [16, 17, 20, 23],
            19: [16, 21, 3, 11],
            20: [12, 4, 18, 23],
            21: [16, 19, 22, 23],
            22: [6, 14, 21, 23],
            23: [18, 20, 21, 22],
        }

        self.bits_map = {
            0: tuple(np.array([(0, 1), (0, 5), (1, 0), (1, 6), (5, 0), (5, 6), (6, 1), (6, 5)]).T),  # LSB
            1: tuple(np.array([(0, 2), (0, 4), (2, 0), (2, 6), (4, 0), (4, 6), (6, 2), (6, 4)]).T),  #
            2: tuple(np.array([(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)]).T),  #
            3: tuple(np.array([(3, 3)]).T)  # MSB
        }
        self.read_bits = {
            0: (0, 1),  # LSB
            1: (0, 2),
            2: (1, 2),
            3: (3, 3)   # MSB
        }

        def int_to_bin_string(self, i):
            if i == 0:
                return "0"
            s = ''
            while i:
                if i & 1 == 1:
                    s = "1" + s
                else:
                    s = "0" + s
                i //= 2
            return s