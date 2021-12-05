#!/usr/bin/env python3

import copy
from typing import Tuple

import gym
import numpy as np

import competitor as comp
import customer
import utils as ut
from customer import Customer

# An offer is a Market State that contains both prices and both qualities


class SimMarket(gym.Env):
	def __init__(self) -> None:
		self.competitors = self.get_competitor_list()
		# The agent's price does not belong to the observation_space any more because an agent should not depend on it
		self.setup_action_observation_space()

		# TODO: Better testing for the observation and action space
		assert (
			self.observation_space and self.action_space
		), 'Your subclass has major problems with setting up the environment'

		# Make sure that variables such as state, customer are known
		self.reset()

	def reset(self) -> np.array:
		self.step_counter = 0

		reset_state = self.reset_agent_information()
		for competitor in self.competitors:
			reset_state += self.reset_competitor_information(competitor)

		self.state = np.array(reset_state)

		self.customer = self.choose_customer()

		# print('I initiate with', self.state)
		return copy.deepcopy(self.state)

	def simulate_customers(self, profits, offers, n) -> None:
		for _ in range(n):
			customer_buy, customer_return = self.customer.buy_object(offers)
			# when using LE-Market, customer_return is None
			if customer_return is not None:
				self.apply_customer_return(customer_return)
			if customer_buy != 0:
				self.complete_purchase(offers, profits, customer_buy)

	def generate_offer(self, action) -> np.array:
		# add agent prices to state array
		return np.concatenate(
			(self.action_to_array(action), self.state), dtype=np.float64
		)

	def modify_profit_by_state(self, profits) -> None:
		pass

	def apply_customer_return(self, customer_return) -> None:
		assert False

	def step(self, action) -> Tuple[np.array, np.float64, bool, dict]:
		# The action is the new price of the agent

		err_msg = '%r (%s) invalid' % (action, type(action))
		assert self.action_space.contains(action), err_msg

		self.step_counter += 1
		n_vendors = (
			len(self.competitors) + 1
		)  # The number of competitors plus the agent

		profits = [0] * n_vendors

		for i in range(n_vendors):
			self.simulate_customers(
				profits,
				self.generate_offer(action),
				int(np.floor(ut.NUMBER_OF_CUSTOMERS / n_vendors)),
			)
			if i < len(self.competitors):
				action_competitor_i = self.competitors[i].give_competitors_price(
					self.generate_offer(action), i + 1
				)
				self.apply_competitor_action(action_competitor_i, i)

		self.modify_profit_by_state(profits)

		output_dict = {'all_profits': profits}
		is_done = self.step_counter >= ut.EPISODE_LENGTH
		return copy.deepcopy(self.state), profits[0], is_done, output_dict


class LinearEconomy(SimMarket):
	def setup_action_observation_space(self) -> None:
		# cell 0: agent's quality, afterwards: odd cells: competitor's price, even cells: competitor's quality
		self.observation_space = gym.spaces.Box(
			np.array([0.0] * (len(self.competitors) * 2 + 1)),
			np.array(
				[ut.MAX_QUALITY]
				+ [ut.MAX_PRICE, ut.MAX_QUALITY] * len(self.competitors)
			),
			dtype=np.float64,
		)

		# one action for every price possible for 0 and MAX_PRICE
		self.action_space = gym.spaces.Discrete(ut.MAX_PRICE)

	def reset_agent_information(self) -> list:
		return [ut.shuffle_quality()]

	def reset_competitor_information(self, competitor) -> list:
		comp_price, comp_quality = competitor.reset()
		return [comp_price, comp_quality]

	def action_to_array(self, action) -> np.array:
		return np.array([action + 1.0])

	def choose_customer(self) -> Customer:
		return customer.CustomerLinear()

	def complete_purchase(self, offers, profits, customer_buy) -> None:
		profits[customer_buy - 1] += (offers[(customer_buy - 1) * 2] - ut.PRODUCTION_PRICE)

	def ith_competitor_index(self, i) -> int:
		return 2 * i + 1

	def apply_competitor_action(self, action, i) -> None:
		self.state[self.ith_competitor_index(i)] = action


class ClassicScenario(LinearEconomy):
	def get_competitor_list(self) -> list:
		return [comp.CompetitorJust2Players()]


class MultiCompetitorScenario(LinearEconomy):
	def get_competitor_list(self) -> list:
		return [
			comp.CompetitorLinearRatio1(),
			comp.CompetitorRandom(),
			comp.CompetitorJust2Players(),
		]


class CircularEconomy(SimMarket):
	# currently monopoly
	def setup_action_observation_space(self) -> None:
		# cell 0: number of products in the used storage, cell 1: number of products in circulation
		self.max_storage = 1e2
		self.observation_space = gym.spaces.Box(np.array([0, 0]), np.array([self.max_storage, 10 * self.max_storage]), dtype=np.float64)
		self.action_space = gym.spaces.Discrete(ut.MAX_PRICE * ut.MAX_PRICE)  # Every pair of actions encoded in one number

	def get_competitor_list(self) -> list:
		return []

	def reset_agent_information(self) -> list:
		return [int(np.random.rand() * self.max_storage), int(5 * np.random.rand() * self.max_storage)]

	def action_to_array(self, action) -> list:
		# cell 0: price for second-hand-product, cell 1: price for new product
		act = [int(np.floor(action / ut.MAX_PRICE)) + 1, int(action % ut.MAX_PRICE) + 1]
		# print("You perform ", act)
		return act

	def choose_customer(self) -> Customer:
		return customer.CustomerCircular()

	def apply_customer_return(self, customer_return) -> None:
		assert customer_return == 1
		# print("A customer returns a product")

		# check if number of products in circulation > 0
		if self.state[1] >= customer_return:
			# check if storage is full
			if self.state[0] < self.max_storage:
				self.state[0] += customer_return
			self.state[1] -= customer_return

	def complete_purchase(self, offers, profits, customer_buy) -> None:
		# print("I want to buy ", customer_buy)
		assert len(profits) == 1
		assert 0 < customer_buy and customer_buy <= 2
		if customer_buy == 1:
			if self.state[0] >= 1:
				# Increase the profit and decrease the storage
				profits[0] += offers[0]
				self.state[0] -= 1
			else:
				# Punish the agent for not having enough second-hand-products
				profits[0] -= 2 * ut.MAX_PRICE
		elif customer_buy == 2:
			profits[0] += offers[1] - ut.PRODUCTION_PRICE
			# One more product is in circulation now
			self.state[1] = min(self.state[1] + 1, 10 * self.max_storage)

	def modify_profit_by_state(self, profits) -> None:
		# print("Your storage cost is ", self.state[0])
		profits[0] -= self.state[0] / 2  # Storage costs per timestep
