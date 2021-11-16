#!/usr/bin/env python3

# rl
# own files
import utils as ut
from sim_market import CircularEconomy

env = CircularEconomy()
profit = 0
is_done = False
print(env)
state = env.reset()


print('The production price: ', ut.PRODUCTION_PRICE, ' maxprice: ', ut.MAX_PRICE, ' max storage: ', env.max_storage)
while not is_done:
    used_in_storage = state[0]
    in_circulation = state[1]
    print(
        'second hand products in circulation:',
        used_in_storage,
        ' products in circulation:',
        in_circulation
    )
    action = input('What do you want to do? ')
    assert action != ''
    state, reward, is_done, _ = env.step(int(action))
    print('Your profit this round is', reward)
    profit += reward
print('Your total profit:', profit)
