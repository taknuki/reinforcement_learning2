from math import sqrt
import numpy as np
from ..domain import State, Action


class ClassicalAgent:
    def __init__(self, name, exp_param):
        self._name = name
        self.exp_param = exp_param
        self.__agent_actions = {}

    @property
    def name(self):
        """ Returns the name of the agent.

        :returns: The name of the agent.
        :rtype: str
        """
        return self._name

    def act(self, state: State):
        """ Returns the next action for the current state.

        :param State state: The current state.
        :returns: The next action.
        :rtype: Action
        """
        next_actions = [
            self._agent_action(state, a) for a in state.allowed_actions]
        p = self.exp_param * sqrt(sum([a.n for a in next_actions]))
        scores = [a.q + p / (1.0 + a.n) for a in next_actions]
        return next_actions[np.argmax(np.array(scores))].action

    def learn(self, root: State):
        trace = []
        state = root
        while not state.is_terminal:
            agent_action = self._agent_action(state, self.act(state))
            trace.append(agent_action)
            state = state.take_action(agent_action.action)
        reward = state.value
        while trace:
            agent_action = trace.pop()
            agent_action.update(reward)

    def _agent_action(self, state: State, action: Action):
        agent_action = self.__agent_actions.get((state.id, action.id), None)
        if agent_action is None:
            agent_action = ClassicalAgentAction(state, action)
            self.__agent_actions[
                (agent_action.state.id, agent_action.action.id)] = agent_action
        return agent_action


class ClassicalAgentAction:
    def __init__(self, state: State, action: Action):
        """

        :param State state: The state.
        :param Action action: The Action.
        """
        self._state = state
        self._action = action
        self._count = 0
        self._value = 0.0
        self._q = 0.0

    @property
    def state(self) -> State:
        """ Returns the state.

        :returns: The state.
        :rtype: State
        """
        return self._state

    @property
    def action(self) -> Action:
        """ Returns the action.

        :returns: The action.
        :rtype: Action
        """
        return self._action

    def update(self, value):
        """ Updates the statistics.

        :param float value: The action value.
        """
        self._count += 1
        self._value += value
        self._q = self.state.player_turn * self._value / self._count

    @property
    def n(self) -> int:
        """ Returns the visit count.

        :returns: The visit count.
        :rtype: int
        """
        return self._count

    @property
    def q(self) ->float:
        """ Returns the action value.

        :returns: The action value.
        :rtype: float
        """
        return self._q
