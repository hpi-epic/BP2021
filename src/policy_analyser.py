#!/usr/bin/env python3

# rl
import math

import torch
from torch.utils.tensorboard import SummaryWriter

import sim_market
import utils as ut
from model import simple_network

# this file is broken, if you do not have 'best_marketplace.dat'
env = sim_market.CircularEconomy()

model = simple_network(env.observation_space.shape[0], env.action_space.n).to('cpu')
model.load_state_dict(
    torch.load('best_marketplace.dat', map_location=torch.device('cpu'))
)

counter = 0
our_profit = 0
is_done = False
state = env.reset()
print('The production price is', ut.PRODUCTION_PRICE)

writer = SummaryWriter()

with torch.no_grad():
    while not is_done:
        action = int(torch.argmax(model(torch.Tensor(state))))
        writer.add_scalar('Example_state/storage_content', state[0], counter)
        writer.add_scalar('Example_state/products_in_circle', state[1], counter)
        writer.add_scalar('Example_action/price_second_hand', math.floor(action / 10) + 1, counter)
        writer.add_scalar('Example_action/price_new', math.floor(action % 10) + 1, counter)
        print(
            'This is the state:',
            state,
            ' and I will do ',
            action,
            'and this is how I estimate it:',
            torch.max(model(torch.Tensor(state))),
        )
        state, reward, is_done, dict = env.step(action)
        print('The agents profit this round is', reward)
        our_profit += reward
        writer.add_scalar('Example_reward/reward', reward, counter)
        writer.add_scalar('Example_reward/reward_cumulated', our_profit, counter)
        counter += 1
print(
    'In total the agent earned',
    our_profit
)
