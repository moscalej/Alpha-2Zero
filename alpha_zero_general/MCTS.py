import math
import numpy as np
from alpha_zero_general.Game import Game
from alpha_zero_general.NeuralNet import NeuralNet
from Nine_Men_Morris_Alpha_2.Game.NMMLogic import Board  # todo remove after debug

# from scipy.special import softmax

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, nnet: NeuralNet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.branch_mem = {}
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.get_game_ended ended for board s
        self.valid_actions = {}  # stores game.get_valid_moves for board s

    def get_action_prob(self, canonical_board: np.ndarray, temp=1) -> np.ndarray:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonical_board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        for i in range(self.args.numMCTSSims):
            self.branch_mem = {'deep': 0, 'b_s': 'ba', "ba": {}, "bb": {}}
            self.search(canonical_board.copy())

        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if False:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = np.array(counts).copy()
        probs = counts / sum(counts)
        return probs

    def search(self, canonical_board, verbose=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonical_board
        """

        state = self.game.string_representation(canonical_board)
        self.branch_mem['deep'] += 1
        branch_mem = self.branch_mem[self.branch_mem['b_s']]
        if state not in branch_mem:
            self.branch_mem[self.branch_mem['b_s']][state] = self.branch_mem['b_s']
        if self.branch_mem['deep'] > 80:
            return - 1e-4

        if state not in self.Es:
            # checks if the game has ended
            # 0  Means the game still continues
            # 1 means this scope has won the game
            # -1  means this scope has loose the game
            self.Es[state] = self.game.get_game_ended(canonical_board, 1)

        if self.Es[state] != 0:
            # This is a End game node the value for the upper
            assert self.Es[state] == -1, f"Impossible to wim, the value was:{self.Es[state]}"
            return -1 * self.Es[state]

        if state not in self.Ps:  # policy state meaning we did not evaluate this before

            # leaf node
            polici_network, v_network = self.nnet.predict(canonical_board)
            valids = self.game.get_valid_moves(canonical_board, 1)
            polici_network_mask = polici_network * valids  # TODO maybe use sparse matrix
            sum_Ps_s = np.sum(polici_network_mask)

            if sum_Ps_s > 0:
                self.Ps[state] = polici_network_mask / sum_Ps_s
            else:
                # if all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is

                print("All valid moves were masked, do workaround.")
                self.Ps[state] = valids / sum(valids)

            self.valid_actions[state] = valids
            self.Ns[state] = 0
            self.branch_mem["end"] = 1
            return - self.args.n_importance * v_network[0]

        valids = self.valid_actions[state]
        cur_best = -float('inf')
        best_act = -1

        next_branch = 'bb' if self.branch_mem['b_s'] == 'ba' else 'ba'

        # pick the action with the highest upper confidence bound
        for action in np.where(valids)[0]:
            if (state, action) in self.Qsa:
                # This provides some exploration if Nsa is small then u will be bigger --> explore less
                Qsa = self.Qsa[(state, action)]
                Ps_a = self.Ps[state][action]
                Ns = self.Ns[state]
                Nsa = self.Nsa[(state, action)]
                next_s, next_player = self.game.get_next_state(canonical_board, 1, action)
                next_s = self.game.get_canonical_form(next_s, next_player)
                next_string = self.game.string_representation(next_s)
                cpuct = self.args.cpuct
                if next_string in self.branch_mem[next_branch]:
                    continue
                b = cpuct * Ps_a * math.sqrt(Ns) / (1 + Nsa)
                u = Qsa + cpuct * Ps_a * math.sqrt(Ns) / (1 + Nsa)

            else:
                Ps_a = self.Ps[state][action]
                Ns = self.Ns[state]
                cpuct = self.args.cpuct

                u = cpuct * Ps_a * math.sqrt(Ns + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = action
                # second_best = best_act
                # fird_best = second_best
        if best_act == -1:
            return 1e-4
        action = best_act

        # <debug>
        if verbose:
            b = self.game.get_board_obj(canonical_board)
            print(f"Player1 step {b.decode_step_count()}:")
            b.verbose_game(canonical_board, action)
        # <\debug>

        next_s, next_player = self.game.get_next_state(canonical_board, 1, action)
        next_s = self.game.get_canonical_form(next_s, next_player)
        try:
            next_s_string = self.game.string_representation(next_s)
            self.branch_mem['b_s'] = next_branch
            v = self.search(next_s)  # a

        except RecursionError:
            print("recursion")
            t = np.array(self.Ps[state])
            print(f"np.sum(self.Ps[s]){np.sum(t != 0)}")
            print(f'Values are : {t[t != 0]}')
            self.Es[state] != 1e-4
            return 0

        if (state, action) in self.Qsa:  # if (s,a) exists, update, otherwise, set

            Nsa = self.Nsa[(state,action)]
            Qsa = self.Qsa[(state,action)]
            quality = (Nsa * Qsa + v) / (Nsa + 1)
            self.Qsa[(state, action)] = quality
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = v
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return - v * 0.95
