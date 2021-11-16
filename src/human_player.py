#!/usr/bin/env python3

# rl
# own files
import utils as ut
from sim_market import ClassicScenario

env = ClassicScenario()
our_profit = 0
is_done = False
state = env.reset()


print('The production price: ', ut.PRODUCTION_PRICE, ' maxprice: ', ut.MAX_PRICE)
while not is_done:
    agent_quality = state[0]
    comp_price = state[1]
    comp_quality = state[2]
    print(
        'agent_quality',
        agent_quality,
        'comp_price:',
        comp_price,
        'comp_quality',
        comp_quality,
    )
    action = input('What do you want to do? ')
    assert action != ''
    state, reward, is_done, _ = env.step(int(action))
    print('Your profit this round is', reward)
    our_profit += reward
print('Your total earnings:', our_profit)
