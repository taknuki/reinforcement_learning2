import numpy as np


class FoolAgent:
    """ This agent selects the next action at random.
    """
    def __init__(self, name):
        """
        :param str name: The name of the agent.
        """
        self._name = name

    @property
    def name(self):
        """ Returns the name of the agent.

        :returns: The name of the agent.
        :rtype: str
        """
        return self._name

    def act(self, state):
        """ Returns the next action for the current state.

        :param State state: The current state.
        :returns: The next action.
        :rtype: Action
        """
        return np.random.choice(state.allowed_actions)
