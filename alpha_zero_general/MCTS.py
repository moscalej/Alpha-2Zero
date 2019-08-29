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
        self.Vs = {}  # stores game.get_valid_moves for board s

    def get_action_prob(self, canonical_board: np.ndarray, temp=1) -> np.ndarray:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonical_board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        b = Board(canonical_board)
        debug = b.decode_step_count()
        for i in range(self.args.numMCTSSims):
            self.branch_mem = {'deep': 0}  # TODO
            self.search(canonical_board.copy())

        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if False:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # counts = [x ** (1. / temp) for x in counts]
        # counts = [x ** (1. / temp) for x in counts]
        # probs = [x / float(sum(counts)) for x in counts]
        # probs = softmax(counts) * np.array(counts) != 0
        counts = np.array(counts)
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
        if state not in self.branch_mem:
            self.branch_mem[state] = 1
        elif self.branch_mem['deep'] > 100:
            return - 1e-4

        else:
            self.branch_mem[state] += 1
            # return - 1
        if state not in self.Es:
            # checks if the game has ended
            # 0  Means the game still continues
            # 1 means this scope has won the game
            # -1  means this scope has loose the game
            self.Es[state] = self.game.get_game_ended(canonical_board, 1)

        if self.Es[state] != 0:
            # This is a End game node the value for the upper

            # scope will be the negative of this one
            # print(f'Monte Carlo find a winner and is :{self.Es[state]}')
            return -self.Es[state]


        if state not in self.Ps:  # policy state meaning we did not evaluate this before

            # leaf node
            self.Ps[state], v_network = self.nnet.predict(canonical_board)
            t = max(self.Ps[state])
            valids = self.game.get_valid_moves(canonical_board, 1)
            self.Ps[state] = self.Ps[state] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[state])

            if sum_Ps_s > 0:
                self.Ps[state] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is
                # insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay
                # attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[state] = self.Ps[state] + valids
                self.Ps[state] /= np.sum(self.Ps[state])

            self.Vs[state] = valids
            self.Ns[state] = 0
            self.branch_mem["end"] = 1
            return -v_network[0]

        valids = self.Vs[state]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for action in range(self.game.get_action_size()):
            if valids[action]:
                if (state, action) in self.Qsa:
                    # This provides some exploration if Nsa is small then u will be bigger --> explore less
                    Qsa = self.Qsa[(state, action)]
                    Ps = self.Ps[state][action]
                    Ns = self.Ns[state]
                    Nsa = self.Nsa[(state, action)]
                    next_s, next_player = self.game.get_next_state(canonical_board, 1, action)
                    next_s = self.game.get_canonical_form(next_s, next_player)
                    next_string = self.game.string_representation(next_s)
                    if next_string in self.branch_mem:
                        u = Qsa + self.args.cpuct * Ps * math.sqrt(Ns) / (1 + Nsa + self.branch_mem[next_string]**2)
                    else:
                        u = Qsa + self.args.cpuct * Ps * math.sqrt(Ns) / (1 + Nsa )
                else:
                    Ps = self.Ps[state][action]
                    Ns = self.Ns[state]
                    u = self.args.cpuct * Ps * math.sqrt(Ns + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = action

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
            v = self.search(next_s)  # a

        except RecursionError:
            print("recursion")
            t = np.array(self.Ps[state])
            print(f"np.sum(self.Ps[s]){np.sum(t != 0)}")
            print(f'Values are : {t[t != 0]}')
            self.Es[state] != 1e-4
            return 0

        if (state, action) in self.Qsa:  # if (s,a) exists, update, otherwise, set
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + v) / (
                    self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = v
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return -v
