from tqdm import trange
import numpy as np
from lib.domain import Game, play_repeat
from lib.agent import FoolAgent, ClassicalAgent, DNNAgent

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    game = Game()
    fool = FoolAgent('alpha')
    classical = ClassicalAgent('beta', 0.5)
    for _ in trange(10000):
        classical.learn(game.initial_state())
    classical.exp_param = 0
    play_repeat(
        game,
        fool,
        classical,
        100
    )

    agent = DNNAgent('gamma', game)
    for i in range(500):
        agent = agent.learn()
        play_repeat(
            game,
            classical,
            agent,
            100
        )
