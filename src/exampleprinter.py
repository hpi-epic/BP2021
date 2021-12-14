#!/usr/bin/env python3

import torch
from torch.utils.tensorboard import SummaryWriter

import agent as a
import sim_market


def print_example(env=sim_market.CircularEconomy(), agent=a.RuleBasedCEAgent()):
	counter = 0
	our_profit = 0
	is_done = False
	state = env.reset()
	# print('The production price is', ut.PRODUCTION_PRICE)
	writer = SummaryWriter()

	with torch.no_grad():
		while not is_done:
			action = agent.policy(state)
			print(state)
			if isinstance(env, sim_market.CircularEconomy):
				writer.add_scalar('Example_state/storage_content', env.state[0], counter)
				writer.add_scalar('Example_state/products_in_circle', env.state[1], counter)
				writer.add_scalar('Example_action/price_second_hand', action[0] + 1, counter)
				writer.add_scalar('Example_action/price_new', action[1] + 1, counter)
				if isinstance(env, sim_market.CircularEconomyRebuyPrice):
					writer.add_scalar('Example_action/rebuy_price', action[2] + 1, counter)
			# elif isinstance(env, sim_market.LinearEconomy):
			# 	writer.add_scalar('Example_state/agent_quality', env.state[0], counter)
			# 	writer.add_scalar('Example_state/competitor_quality', env.state[2], counter)
			# 	writer.add_scalar('Example_state/competitor_price', env.state[1], counter)
			# 	writer.add_scalar('Example_action/price_agent', action + 1, counter)
			# print(
			# 	'This is the state:',
			# 	env.state,
			# 	' and I will do ',
			# 	action
			# )
			state, reward, is_done, dict = env.step(action)
			# print('The agents profit this round is', reward)
			our_profit += reward
			# writer.add_scalar('Example_reward/reward', reward, counter)
			# writer.add_scalar('Example_reward/reward_cumulated', our_profit, counter)
			counter += 1
	# print(
	# 	'In total the agent earned',
	# 	our_profit
	# )
	return our_profit


def main():
	agent = a.HumanPlayer
	environment = sim_market.ClassicScenario()
	cumulative_reward = 0
	for i in range(1000):
		cumulative_reward += print_example(environment, agent)

	print('The average reward is', cumulative_reward / 1000)


if __name__ == '__main__':
	agent = a.HumanPlayer()
	environment = sim_market.ClassicScenario()
	print_example(environment, agent)
