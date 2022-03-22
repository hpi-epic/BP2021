from abc import ABC

import numpy as np

from recommerce.configuration.hyperparameter_config import config
from recommerce.market.vendors import Agent, FixedPriceAgent, HumanPlayer, RuleBasedAgent


class CircularAgent(Agent, ABC):
	pass


class HumanPlayerCE(CircularAgent, HumanPlayer):
	def __init__(self, name='YOU - Circular'):
		self.name = name
		print('Welcome to this funny game! Now, you are the one playing the game!')

	def policy(self, observation, *_) -> tuple:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 1, 'Please enter two numbers seperated by spaces!'
		price_old, price_new = raw_input_string.split(' ')
		return (int(price_old), int(price_new))


class HumanPlayerCERebuy(HumanPlayerCE):
	def policy(self, observation, *_) -> tuple:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 2, 'Please enter three numbers seperated by spaces!'
		price_old, price_new, rebuy_price = raw_input_string.split(' ')
		return (int(price_old), int(price_new), int(rebuy_price))


class FixedPriceCEAgent(CircularAgent, FixedPriceAgent):
	def __init__(self, fixed_price=(2, 4), name='fixed_price_ce'):
		assert isinstance(fixed_price, tuple), f'fixed_price must be a tuple: {fixed_price} ({type(fixed_price)})'
		assert len(fixed_price) == 2, f'fixed_price must contain two values: {fixed_price}'
		assert all(isinstance(price, int) for price in fixed_price), f'the prices in fixed_price must be integers: {fixed_price}'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> tuple:
		return self.fixed_price


class FixedPriceCERebuyAgent(FixedPriceCEAgent):
	def __init__(self, fixed_price=(3, 6, 2), name='fixed_price_ce_rebuy'):
		assert isinstance(fixed_price, tuple), f'fixed_price must be a tuple: {fixed_price} ({type(fixed_price)})'
		assert len(fixed_price) == 3, f'fixed_price must contain three values: {fixed_price}'
		assert all(isinstance(price, int) for price in fixed_price), f'the prices in fixed_price must be integers: {fixed_price}'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> tuple:
		return self.fixed_price


class RuleBasedCEAgent(RuleBasedAgent, CircularAgent):
	def __init__(self, name='rule_based_ce'):
		self.name = name

	def return_prices(self, price_old, price_new, rebuy_price):
		return (price_old, price_new)

	def policy(self, observation, epsilon=0) -> tuple:
		# this policy sets the prices according to the amount of available storage
		products_in_storage = observation[1]
		price_old = 0
		price_new = config.production_price
		rebuy_price = 0
		if products_in_storage < config.max_storage / 15:
			# fill up the storage immediately
			price_old = int(config.max_price * 6 / 10)
			price_new += int(config.max_price * 6 / 10)
			rebuy_price = price_old - 1

		elif products_in_storage < config.max_storage / 10:
			# fill up the storage
			price_old = int(config.max_price * 5 / 10)
			price_new += int(config.max_price * 5 / 10)
			rebuy_price = price_old - 2

		elif products_in_storage < config.max_storage / 8:
			# storage content is ok
			price_old = int(config.max_price * 4 / 10)
			price_new += int(config.max_price * 4 / 10)
			rebuy_price = price_old // 2
		else:
			# storage too full, we need to get rid of some refurbished products
			price_old = int(config.max_price * 2 / 10)
			price_new += int(config.max_price * 7 / 10)
			rebuy_price = 0

		price_new = min(9, price_new)
		assert price_old <= price_new, 'The price for used products should be lower or equal to the price of new products'
		return self.return_prices(price_old, price_new, rebuy_price)


class RuleBasedCERebuyAgent(RuleBasedCEAgent):
	def return_prices(self, price_old, price_new, rebuy_price):
		return (price_old, price_new, rebuy_price)


class RuleBasedCERebuyAgentCompetitive(RuleBasedAgent, CircularAgent):
	def __init__(self, name='rule_based_ce_rebuy_competitive'):
		self.name = name

	def _clamp_price(self, price, min_price=0, max_price=config.max_price - 1) -> int:
		price = int(price)
		price = max(price, min_price)
		price = min(price, max_price)
		return price

	def policy(self, observation, *_) -> tuple:
		assert isinstance(observation, np.ndarray), 'observation must be a np.ndarray'
		assert len(observation) == 6, 'observation is made for this agent and one competitor -> length must be 6'

		# in_circulation is ignored
		own_storage = observation[1].item()
		competitor_old_price = observation[2].item()
		competitor_new_price = observation[3].item()
		competitor_rebuy_price = observation[4].item()

		price_new = max(competitor_new_price - 1, config.production_price + 1)
		# competitor's storage is ignored
		if own_storage < config.max_storage / 15:
			# fill up the storage immediately
			price_old = competitor_old_price + 2
			rebuy_price = min(competitor_rebuy_price + 1, 2)
		elif own_storage < config.max_storage / 10:
			# fill up the storage
			price_old = competitor_old_price + 1
			rebuy_price = competitor_rebuy_price
		elif own_storage < config.max_storage / 8:
			# storage content is ok
			rebuy_price = max(competitor_rebuy_price - 1, 1)
			price_old = max(competitor_old_price - 1, rebuy_price + 1)
		else:
			# storage too full, we need to get rid of some refurbished products
			rebuy_price = max(competitor_rebuy_price - 2, 1)
			price_old = max(competitor_old_price - 2, rebuy_price + 1)

		return (self._clamp_price(price_old), self._clamp_price(price_new), self._clamp_price(rebuy_price))
