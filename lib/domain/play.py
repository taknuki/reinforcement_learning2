import random
from tqdm import trange
from . import Game


def play_repeat(game: Game, agent1, agent2, times):
    active = random.choice([-1, 1])
    agents = {
        active: agent1,
        active * -1: agent2,
    }
    result = {
        agent1.name: 0,
        'tie': 0,
        agent2.name: 0,
    }
    for _ in trange(times):
        state = game.initial_state()
        active_agent = 1
        while True:
            action = agents[active_agent].act(state)
            state = state.take_action(action)
            if state.is_tie:
                result['tie'] += 1
                break
            if state.is_terminal:
                result[agents[active_agent].name] += 1
                break
            active_agent *= -1
    print('{0} win {1}, {2} win {3}, tie {4}'.format(
        agent1.name, result[agent1.name], agent2.name, result[agent2.name], result['tie']))


def play(game: Game, agent1, agent2, logger):
    agents = {
        1: agent1,
        -1: agent2,
    }
    active_agent = 1
    state = game.initial_state()
    while True:
        action = agents[active_agent].act(state)
        state.take_action(action)
        print(action)
        state.render(logger)
        if state.is_tie:
            print('tie')
            break
        if state.is_terminal:
            print('{0} win'.format(agents[active_agent].name))
            break
        active_agent *= -1
