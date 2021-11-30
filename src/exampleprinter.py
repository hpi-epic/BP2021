#!/usr/bin/env python3

import math

import torch
from torch.utils.tensorboard import SummaryWriter

import agent as a
import sim_market

# import utils as ut

situation = 'circular'

env = sim_market.CircularEconomy() if situation == 'circular' else sim_market.ClassicScenario()
agent = a.FixedPriceAgent(67)  # a.RuleBasedCEAgent()  # a.HumanPlayer() #a.QLearningAgent(env.observation_space.shape[0], env.action_space.n, load_path='trainedModels\\BITTE PFAD EINGEBEN')  # a.HumanPlayer() if you want to play the game

counter = 0
our_profit = 0
is_done = False
state = env.reset()
# print('The production price is', ut.PRODUCTION_PRICE)
print('agent action', agent.policy(state))

writer = SummaryWriter()

with torch.no_grad():
    while not is_done:
        action = agent.policy(state)
        if situation == 'circular':
            writer.add_scalar('Example_state/storage_content', env.state[0], counter)
            writer.add_scalar('Example_state/products_in_circle', env.state[1], counter)
            writer.add_scalar('Example_action/price_second_hand', math.floor(action / 10) + 1, counter)
            writer.add_scalar('Example_action/price_new', math.floor(action % 10) + 1, counter)
        elif situation == 'linear':
            writer.add_scalar('Example_state/agent_quality', env.state[0], counter)
            writer.add_scalar('Example_state/competitor_quality', env.state[2], counter)
            writer.add_scalar('Example_state/competitor_price', env.state[1], counter)
            writer.add_scalar('Example_action/price_agent', action + 1, counter)
        print(
            'This is the state:',
            env.state,
            ' and I will do ',
            action
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
