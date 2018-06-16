from typing import Tuple
import numpy as np

PIECES = {'1': 'X', '0': '-', '-1': 'O'}


class Game:
    @property
    def name(self) -> str:
        return 'OX Game'

    @property
    def state_shape(self) -> Tuple[int, int, int]:
        """ Returns the shape of the state.

        :returns: The shape of the state
        """
        return 2, 3, 3

    @property
    def action_size(self) -> int:
        """ Returns the size of action space.

        :returns: The size of action space.
        """
        return 9

    @staticmethod
    def initial_state():
        return State(
            board=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int),
            player_turn=1)


class State:
    _winners = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    def __init__(self, board, player_turn):
        """
        :param np.ndarray board: The game board.
        :param int player_turn: 1 or -1
        """
        self._board = board
        self._player_turn = player_turn
        # below members are property
        self._id = ''.join([PIECES[str(i)] for i in board])
        self._binary = self.__binary()
        for winner in self._winners:
            if sum([self._board[x] for x in winner]) == 3 * -self._player_turn:
                self._is_terminal = True
                self._is_tie = False
                break
        else:
            if np.count_nonzero(self._board) == len(self._board):
                self._is_terminal = True
                self._is_tie = True
            else:
                self._is_terminal = False
                self._is_tie = False
        if self._is_terminal and not self._is_tie:
            self._value = -self._player_turn
        else:
            self._value = 0
        if self._is_terminal:
            self._allowed_actions = []
        else:
            self._allowed_actions = [
                Action(i) for i in range(len(board)) if board[i] == 0]

    @property
    def id(self):
        return self._id

    @property
    def player_turn(self):
        return self._player_turn

    @property
    def allowed_actions(self):
        return self._allowed_actions

    @property
    def value(self):
        return self._value

    @property
    def is_tie(self):
        return self._is_tie

    @property
    def is_terminal(self):
        return self._is_terminal

    def take_action(self, action):
        """ Takes the next action and transit to the next state.

        :param Action action: The action.
        :returns: The next state.
        :rtype: State
        """
        new_board = np.array(self._board)
        new_board[action.id] = self._player_turn
        return State(new_board, -self._player_turn)

    def render(self):
        """ Renders the game state.
        """
        print('--------------')
        for r in range(3):
            print(
                [PIECES[str(x)] for x in self._board[3*r: (3*r + 3)]])
        print('--------------')

    @property
    def binary(self):
        return self._binary

    def __binary(self):
        ret = np.zeros((2, len(self._board)), dtype=np.int)
        for i, p in enumerate(self._board):
            ret[0][i] = 1 if p == self._player_turn else 0
            ret[1][i] = 1 if p == -self._player_turn else 0
            # ret[2][i] = 1 if 1 == self._player_turn else 0
        return np.reshape(ret, (2, 3, 3))


class Action:
    def __init__(self, position):
        self._position = position

    @property
    def id(self) -> int:
        """ Returns the id of the Action.

        :returns: The id.
        :rtype: int
        """
        return self._position
