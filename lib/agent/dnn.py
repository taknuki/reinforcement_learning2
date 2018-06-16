from math import sqrt
import random
from tqdm import trange
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, LeakyReLU
from keras.layers import Flatten, BatchNormalization, add
from keras import regularizers
from keras.optimizers import SGD
import tensorflow as tf
from ..domain import Game, State, Action


def softmax_cross_entropy_with_logits(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)


class ResCNN:
    def __init__(
            self,
            reg_const,
            learning_rate,
            momentum,
            input_dim,
            output_dim,
            hidden_layers,
    ):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model: Model = self._build()

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, tgts, epochs, verbose, validation_split, batch_size):
        return self.model.fit(
            states,
            tgts,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            batch_size=batch_size,
        )

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def conv_layer(self, input_block, filters, kernel_size):
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(input_block)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x

    def residual_layer(self, input_block, filters, kernel_size):
        x = self.conv_layer(input_block, filters, kernel_size)
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)
        x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x

    def value_head(self, input_block):
        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(input_block)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            20,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)
        x = LeakyReLU()(x)
        x = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='value_head',
        )(x)
        return x

    def policy_head(self, input_block):
        x = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(input_block)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            self.output_dim,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='policy_head',
        )(x)
        return x

    def _build(self) -> Model:
        main_input = Input(shape=self.input_dim, name='main_input')
        x = self.conv_layer(
            main_input,
            self.hidden_layers[0]['filters'],
            self.hidden_layers[0]['kernel_size'],
        )
        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)
        model = Model(
            inputs=[main_input],
            outputs=[vh, ph],
        )
        model.compile(
            loss={
                'value_head': 'mean_squared_error',
                'policy_head': softmax_cross_entropy_with_logits,
            },
            optimizer=SGD(
                lr=self.learning_rate,
                momentum=self.momentum,
            ),
            loss_weights={
                'value_head': 0.5,
                'policy_head': 0.5
            },
        )
        return model


class DNNAgentNNProvider:
    def __init__(self, game: Game):
        self._state_shape = game.state_shape
        self._action_size = game.action_size

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    def generate(self, original: ResCNN = None) -> ResCNN:
        nn = ResCNN(
            reg_const=0.0001,
            learning_rate=0.1,
            momentum=0.9,
            input_dim=self.state_shape,
            output_dim=self.action_size,
            hidden_layers=[
                {'filters': 75, 'kernel_size': (4, 4)},
                {'filters': 75, 'kernel_size': (4, 4)},
                {'filters': 75, 'kernel_size': (4, 4)},
                {'filters': 75, 'kernel_size': (4, 4)},
                {'filters': 75, 'kernel_size': (4, 4)},
                {'filters': 75, 'kernel_size': (4, 4)}
            ],
        )
        if original is not None:
            nn.set_weights(original.get_weights())
        return nn


class DNNAgent:
    def __init__(self, name: str, game: Game, version: int = 0, nn: ResCNN=None):
        self._name = name
        self._version = version
        self._game = game
        self._nn_provider = DNNAgentNNProvider(game)
        if nn is None:
            self._nn = self._nn_provider.generate()
            self._next_nn = self._nn_provider.generate()
        else:
            self._nn = nn
            # set when promoted.
            self._next_nn = None
        self._mcts_cycle: int = 50
        self._mcts = DNNAgentMCTS(50, self._nn, 1.0)
        self._tau: float = 1.0
        self._learning_cycle: int = 10
        self._learning_size: int = 128
        self._num_of_eval_games: int = 50
        self._num_of_learning_games: int = 100
        self._compete_threshold: float = 0.55
        self._play_log = PlayLog()

    @property
    def name(self) -> str:
        return '{0}@{1}'.format(self._name, self._version)

    @property
    def version(self) -> int:
        return self._version

    @property
    def game(self) -> Game:
        return self._game

    @property
    def mcts_cycle(self):
        return self._mcts_cycle

    @property
    def tau(self):
        return self._tau

    @property
    def learning_cycle(self):
        return self._learning_cycle

    @property
    def learning_size(self):
        return self._learning_size

    @property
    def num_of_eval_games(self):
        return self._num_of_eval_games

    @property
    def num_of_learning_games(self):
        return self._num_of_learning_games

    @property
    def compete_threshold(self):
        return self._compete_threshold

    def act(self, state: State) -> Action:
        """ Selects the next action.

        :param State state: The current game state.
        :returns: The next action.
        :rtype: Action
        """
        ns = self._mcts.search(state)
        actions = self._mcts.agent_state(state).actions
        if self.tau <= 0.0:
            pi = np.zeros(len(actions), dtype=np.float)
            argmax = np.argmax(ns)
            pi[argmax] = 1.0
        else:
            pi = np.power(ns, 1.0 / self.tau)
            s = np.sum(pi)
            pi = pi / s
            argmax = np.random.choice(len(actions), p=pi)
        pi_all = np.zeros(self._game.action_size, dtype=np.float32)
        for i, a in enumerate(actions):
            pi_all[a.action.id] = pi[i]
        self._play_log.add_state(state.binary, pi_all)
        # print(self.name)
        # self._mcts.agent_state(state).render()
        # print(ns)
        # print(pi_all)
        return actions[argmax].action

    def learn(self):
        self._generate_play_log()
        candidate = self._generate_candidate()
        return self._compete(candidate)

    def _promote(self):
        self._nn.model.save('./dat/{0}.h5'.format(self.name))
        if self._next_nn is None:
            self._next_nn = self._nn_provider.generate(self._nn)
        return self

    def _generate_play_log(self):
        """Generates the self play log.
        """
        self._play_log.reset()
        for _ in trange(self.num_of_learning_games):
            state = self.game.initial_state()
            self._tau = 1.0
            count = 1
            while not state.is_terminal:
                action = self.act(state)
                state = state.take_action(action)
                count += 1
                if count > 3:
                    self._tau = 0.0
            reward = state.value
            self._play_log.commit(reward)

    def _generate_candidate(self):
        candidate = DNNAgent(self._name, self._game, self.version + 1, self._next_nn)
        for _ in range(self.learning_cycle):
            sample = self._play_log.sample(self.learning_size)
            states = np.array([row[0] for row in sample])
            targets = {
                'value_head': np.array([row[2] for row in sample]),
                'policy_head': np.array([row[1] for row in sample])}
            candidate._nn.fit(states, targets, epochs=1, verbose=0, validation_split=0, batch_size=32)
        return candidate

    def _compete(self, candidate: 'DNNAgent'):
        backup = self._mcts
        self._mcts = DNNAgentMCTS(self.mcts_cycle, self._nn, 1.0)
        self._tau = 0.0
        candidate._tau = 0.0
        sb = ScoreBoard()
        for _ in trange(self.num_of_eval_games):
            state = self.game.initial_state()
            active = random.choice([-1, 1])
            players = {
                active: self,
                active * -1: candidate,
            }
            while not state.is_terminal:
                action = players[state.player_turn].act(state)
                state = state.take_action(action)
            if state.is_tie:
                sb.tie()
            else:
                if state.value == active:
                    sb.lose()
                else:
                    sb.win()
        # sb.render()
        if sb.ratio > self.compete_threshold:
            return candidate._promote()
        else:
            self._mcts = backup
            return self


class DNNAgentMCTS:
    """ DNNAgentMCTS is the MCTS unit of the DNNAgent.
    """
    def __init__(self, cycle, nn, c):
        """
        :param int cycle: The number of MCTS search cycle.
        :param ResCNN nn: The neural network.
        :param float c: The constant determining the level of exploration.
        """
        self._cycle = cycle
        self._nn = nn
        self._c = c
        self._agent_states = {}

    @property
    def cycle(self):
        return self._cycle

    def search(self, root):
        """ Search the MCTS tree.

        :param State root: the root node of the MCTS.
        :returns: The search result.
        :rtype: np.ndarray
        """
        root_state = self.agent_state(root)
        if root_state is None:
            before = None
        else:
            before = np.array([a.n for a in root_state.actions])
        for _ in range(self.cycle):
            self.run(root)
        after = np.array([a.n for a in self.agent_state(root).actions])
        if before is None:
            return after
        else:
            return after - before

    def run(self, root):
        """ Run the MCTS.

        :param State root: The root node of the MCTS.
        """
        trace = []
        state = root
        agent_state = self.agent_state(state)
        is_first_move = True
        while not (state.is_terminal or agent_state is None):
            if is_first_move:
                agent_action = self.select_first(agent_state)
                is_first_move = False
            else:
                agent_action = self.select(agent_state)
            trace.append(agent_action)
            state = state.take_action(agent_action.action)
            agent_state = self.agent_state(state)
        if agent_state is None:
            agent_state = self.expand(state)

        # The action statistics are updated in a backward pass through.
        value = agent_state.value
        for agent_action in trace:
            agent_action.update(value)

    def agent_state(self, state):
        """ Returns the corresponding DNNAgentState.
        If not expanded yet, returns None.

        :param State state: The State.
        :returns: The corresponding state.
        :rtype: DNNAgentState
        """
        return self._agent_states.get(state.id, None)

    def select(self, state):
        """ Selects the next action,
        according to the Upper Confidence Bound.

        :param DNNAgentState state: The current game state.
        :returns: The next action.
        :rtype: DNNAgentAction
        """
        next_actions = state.actions
        p = self._c * sqrt(sum([a.n for a in next_actions]))
        scores = [a.q + p * a.p / (1.0 + a.n) for a in next_actions]
        return next_actions[np.argmax(np.array(scores))]

    def select_first(self, state):
        """ Selects the first action,
        according to the Upper Confidence Bound
        and Dirichlet noise.

        :param DNNAgentState state: The current game state.
        :returns: The next action.
        :rtype: DNNAgentAction
        """
        epsilon = 0.2
        alpha = 0.03
        next_actions = state.actions
        dns = np.random.dirichlet([alpha] * len(next_actions))
        p = self._c * sqrt(sum([a.n for a in next_actions]))
        scores = [
            a.q + p * ((1.0 - epsilon) * a.p + epsilon * dn) / (1.0 + a.n)
            for a, dn in zip(next_actions, dns)]
        return next_actions[np.argmax(np.array(scores))]

    def expand(self, state):
        """ Expands the next state using the neural network prediction.

        :param State state: The state.
        :return: The state.
        :rtype: DNNAgentState
        """
        result = self._nn.predict(np.array([state.binary]))
        value = result[0][0][0]
        policy = result[1][0]
        agent_state = DNNAgentState(state, value, policy)
        self._agent_states[state.id] = agent_state
        return agent_state


class DNNAgentState:
    def __init__(self, state, value, policy_logits):
        """
        :param State state: The state.
        :param float value: The value of the state.
        :param list[float] policy_logits: The list of probability logits of next actions.
        """
        self._state = state
        if state.is_terminal:
            self._value = state.value
            self._actions = []
        else:
            self._value = value
            actions = state.allowed_actions
            policy = [policy_logits[a.id] for a in actions]
            policy = np.exp(policy)
            _sum = np.sum(policy)
            policy = policy / _sum
            self._actions = [
                DNNAgentAction(state, a, p) for a, p in zip(actions, policy)]

    @property
    def state(self):
        return self._state

    @property
    def value(self):
        return self._value

    @property
    def actions(self):
        """ Returns the allowed actions from this state.

        :returns: the list of allowd actions.
        :rtype: list(DNNAgentAction)
        """
        return self._actions

    def render(self):
        self.state.render()
        print(' '.join(['{0:.2f}'.format(a.q) for a in self.actions]))
        print(' '.join(['{0:.2f}'.format(a.p) for a in self.actions]))
        print(['{0}'.format(a.n) for a in self.actions])


class DNNAgentAction:
    def __init__(self, state, action, probability):
        """
        :param State state: The current state.
        :param Action action: The next Action.
        :param float probability: The probability of choosing this action.
        """
        self._state = state
        self._player_turn = state.player_turn
        self._action = action
        self._count = 0
        self._value = 0.0
        self._q = 0.0
        self._probability = probability

    @property
    def state(self):
        return self._state

    @property
    def player_turn(self):
        return self._player_turn

    @property
    def action(self):
        return self._action

    @property
    def n(self):
        """ Returns the visit count.

        :returns: The visit count.
        :rtype: int
        """
        return self._count

    @property
    def q(self):
        """ Returns the action value.

        :returns: The action value.
        :rtype: float
        """
        return self._q

    @property
    def p(self):
        """ Returns the probability.

        :returns: The probability of choosing this action.
        :rtype: float
        """
        return self._probability

    def update(self, value):
        self._count += 1
        self._value += value
        self._q = self.player_turn * self._value / self._count


class PlayLog:
    def __init__(self):
        self._play_log = []
        self._state_log = []

    def add_state(self, state, pi):
        self._state_log.append((state, pi))

    def commit(self, reward):
        self._play_log.extend([
            (l[0], l[1], reward) for l in self._state_log
        ])
        self._state_log = []

    def reset(self):
        self._state_log = []
        self._play_log = []

    def sample(self, size):
        if size > len(self._play_log):
            return self._play_log
        return random.sample(self._play_log, size)


class ScoreBoard:
    def __init__(self):
        self._board = {'win': 0, 'tie': 0, 'lose': 0}

    def _count(self, key):
        self._board[key] += 1

    def win(self):
        self._count('win')

    def tie(self):
        self._count('tie')

    def lose(self):
        self._count('lose')

    @property
    def ratio(self):
        if self._board['win'] + self._board['lose'] == 0:
            return 0.5
        return float(self._board['win']) / float(self._board['win'] + self._board['lose'])

    def render(self):
        print(' '.join([k + ": " + str(v) for k, v in self._board.items()]))
