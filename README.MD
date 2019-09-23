# Alpha Zero for Nine Mem Morrys
## Introduction
DeepMind’s AlphaZero publication is considered a significant landmark in RL (Reinforcement Learning) for board game play. The algorithm achieved superhuman performance in chess, shogi, and go, each with under 24 hours of self-play, using almost no specialized or hard coded human knowledge of the games other than the rules.

We want to replicate and extend their results in novel ways, discovering in the process how various choices affect the performance of the algorithm. The fact that we didn't implement an AlphaZero like algorithm before is very limiting with respect to suggesting novelties. 


## Related Work
There are a few implemented examples of AlphaZero for different games like Othello, Tic Tac Toe, these games have a finite number of movements and states can’t be revisited (no movement backwards). We searched for a game without a public solution with AlphaZero. This comes to be Nine Men Morris, later on we realize that this game adds a degree of complexity that we did not expect. 
Nine Men Morris (NMM)
Nine men's morris is a strategy board game for two players, dating at least to the Roman Empire. The game is also known as Nine Men Morris, mill, mills, the mill game, merels, merrills, merelles, marelles, morelles, and ninepenny marl in English. Nine men morris is a solved game, that is, a game whose optimal strategy has been calculated by humans. It has been shown that with perfect play from both players, the game results in a draw.[3] Its name derives from the Latin word merellus, which means a counter or gaming piece.

## Rules 
The board consists of a grid with twenty-four intersections or points. Each player has nine pieces, or "men", usually coloured black and white. Players try to form 'mills'—three of their own men lined horizontally or vertically—allowing a player to remove an opponent's man from the game. A player wins by reducing the opponent to two pieces (where they could no longer form mills and thus be unable to win), or by leaving them without a legal move, blocked.
The game proceeds in three phases:
Placing men on vacant points
Moving men to adjacent points
(optional phase) Moving men to any vacant point when the player has been reduced to three men

Phase 1: Placing pieces
The game begins with an empty board. The players determine who plays first, then take turns placing their men one per play on empty points. If a player is able to place three of their pieces on contiguous points in a straight line, vertically or horizontally, they have formed a mill and may remove one of their opponent's pieces from the board and the game, with the caveat that a piece in an opponent's mill can only be removed if no other pieces are available. After all men have been placed, phase two begins.
Phase 2: Moving pieces
Players continue to alternate moves, this time moving a piece to an adjacent point. A piece may not "jump" another piece. Players continue to try to form mills and remove their opponent's pieces as in phase one. A player can "break" a mill by moving one of his pieces out of an existing mill, then moving it back to form the same mill a second time (or any number of times), each time removing one of his opponent's men. The act of removing an opponent's man is sometimes called "pounding" the opponent.


## NMM Python 
We create a game that can emulate the rules and stages of Nine Men Morris, this game for each state can provide a set of valid moves, do the actions and can return the next state, we also made a visual representation of the game this way the debug process and understanding of the learning can be done.


## Nine Men Morris and Alpha Zero
This game consists of two stages one will be called the set stage where each player can set their own pieces and the second one consists on the Move part where each player has the ability to move their pieces in order to remove their opponent pieces or block them.  In this section we will explain our state representation and  how we define our action space representation. 
State representation 
We encode the game in a matrix of shape 7×7  using the valid places of the matrix for piece placement, meaning the game uses 24 places for the pieces and each player gets a symbol where player one is 1 and player 2 is -1.  Each player at his turn plays with a canonical form of the board as if he is player 1.

The other 25 places on the matrix are used to encode the step count of the game, until stage 2 (then count is meaningless).
By doing this encoding we manage to express a greater amount of information on a small matrix and still preserve distance relations (closer pieces remain closer) and preserve the boards symmetry to rotations. Taking advantage of the fact that the 5 first steps can’t be confused as a board state in second phase, we have 13 steps to count using the encoding, for which 4 binary bits of information are enough, we encode this count into the board (more details in code under Base_mill class).

## Objectives
- [x] Implement Nine mem Morris 
- [x] Implement the Alpha Zero Algorithm 
- [x] Create a training flow with the game
- [x] Improve the MCTS
- [ ] Improve the DNN
- [x] Find an opponent
- [x] Create an Interface to play against the opponent.
- [x] Beat the Opponent
# Authors 
Alejandro Moscoso (moscalej@gmail.com)  
Alex Finkelshtein ()
