#!/usr/bin/env python3

# rl
# import tensorboard
import torch
import torch.nn as nn

# own files
from sim_market import SimMarket

# this file is broken, if you do not have 'best_marketplace.dat'

model = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 28)
).to('cpu')
model.load_state_dict(
    torch.load('best_marketplace.dat', map_location=torch.device('cpu'))
)

env = SimMarket()
our_profit = 0
is_done = False
state = env.reset(False)
print('The production price is', env.production_price)
while not is_done:
    action = int(torch.argmax(model(torch.Tensor(state))))
print(
    'This is the state:',
    state,
    'and this is how I estimate the actions:',
    model(torch.Tensor(state)),
    ' so I do',
    action,
)
state, reward, is_done, _ = env.step(action)
print('The agents profit this round is', reward)
our_profit += reward
print(
    'In total the agent earned',
    our_profit,
    'with a profit/quality of:',
    round(our_profit / state[1], 3),
    ' and his competitor',
    env.comp_profit,
    'with a profit/quality of:',
    round(env.comp_profit / state[2], 3),
)
