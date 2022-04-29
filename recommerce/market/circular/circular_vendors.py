from abc import ABC
from statistics import median

import numpy as np

from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.market.vendors import Agent, FixedPriceAgent, HumanPlayer, RuleBasedAgent


class CircularAgent(Agent, ABC):
	def _clamp_price(self, price, min_price=0, max_price=None) -> int:

		max_price = self.config.max_price - 1 if max_price is None else max_price
		price = int(price)
		price = max(price, min_price)
		price = min(price, max_price)
		return price

	def _get_competitor_prices(self, observation: np.ndarray, is_rebuy_economy: bool):
		# in_circulation is ignored
		competitors_refurbished_prices = []
		competitors_new_prices = []
		competitors_rebuy_prices = []
		for competitor in range(2, observation.size, 4):
			competitors_refurbished_prices.append(observation[competitor].item())
			competitors_new_prices.append(observation[competitor + 1].item())
			if is_rebuy_economy is True:
				competitors_rebuy_prices.append(observation[competitor + 2].item())
		if is_rebuy_economy is True:
			return competitors_refurbished_prices, competitors_new_prices, competitors_rebuy_prices
		else:
			return competitors_refurbished_prices, competitors_new_prices


class HumanPlayerCE(CircularAgent, HumanPlayer):
	def __init__(self, config=None, name='YOU - Circular'):
		self.name = name
		print('Welcome to this funny game! Now, you are the one playing the game!')

	def policy(self, observation, *_) -> tuple:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 1, 'Please enter two numbers seperated by spaces!'
		price_refurbished, price_new = raw_input_string.split(' ')
		return (int(price_refurbished), int(price_new))


class HumanPlayerCERebuy(HumanPlayerCE):
	def policy(self, observation, *_) -> tuple:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 2, 'Please enter three numbers seperated by spaces!'
		price_refurbished, price_new, rebuy_price = raw_input_string.split(' ')
		return (int(price_refurbished), int(price_new), int(rebuy_price))


class FixedPriceCEAgent(CircularAgent, FixedPriceAgent):
	"""
	This vendor's policy is trying to succeed with set constant prices.
	"""
	def __init__(self, config: HyperparameterConfig=None, fixed_price=(2, 4), name='fixed_price_ce'):
		assert isinstance(fixed_price, tuple), f'fixed_price must be a tuple: {fixed_price} ({type(fixed_price)})'
		assert len(fixed_price) == 2, f'fixed_price must contain two values: {fixed_price}'
		assert all(isinstance(price, int) for price in fixed_price), f'the prices in fixed_price must be integers: {fixed_price}'
		self.config = config
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> tuple:
		return self.fixed_price


class FixedPriceCERebuyAgent(FixedPriceCEAgent):
	"""
	This vendor's policy is the a version of the FixedPriceCEAgent with rebuy price.
	"""
	def __init__(self, config: HyperparameterConfig=None, fixed_price=(3, 6, 2), name='fixed_price_ce_rebuy'):
		assert isinstance(fixed_price, tuple), f'fixed_price must be a tuple: {fixed_price} ({type(fixed_price)})'
		assert len(fixed_price) == 3, f'fixed_price must contain three values: {fixed_price}'
		assert all(isinstance(price, int) for price in fixed_price), f'the prices in fixed_price must be integers: {fixed_price}'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> tuple:
		return self.fixed_price


class RuleBasedCEAgent(RuleBasedAgent, CircularAgent):
	"""
	This vendor's policy does not consider the competitor's prices.
	It tries to succeed by taking its own storage costs into account.
	"""
	def __init__(self, config: HyperparameterConfig, name='rule_based_ce'):
		self.name = name
		self.config = config

	def convert_price_format(self, price_refurbished, price_new, rebuy_price):
		return (price_refurbished, price_new)

	def policy(self, observation, epsilon=0) -> tuple:
		# this policy sets the prices according to the amount of available storage
		products_in_storage = observation[1]
		price_refurbished = 0
		price_new = self.config.production_price
		rebuy_price = 0
		if products_in_storage < self.config.max_storage / 15:
			# fill up the storage immediately
			price_refurbished = int(self.config.max_price * 6 / 10)
			price_new += int(self.config.max_price * 6 / 10)
			rebuy_price = price_refurbished - 1

		elif products_in_storage < self.config.max_storage / 10:
			# fill up the storage
			price_refurbished = int(self.config.max_price * 5 / 10)
			price_new += int(self.config.max_price * 5 / 10)
			rebuy_price = price_refurbished - 2

		elif products_in_storage < self.config.max_storage / 8:
			# storage content is ok
			price_refurbished = int(self.config.max_price * 4 / 10)
			price_new += int(self.config.max_price * 4 / 10)
			rebuy_price = price_refurbished // 2
		else:
			# storage too full, we need to get rid of some refurbished products
			price_refurbished = int(self.config.max_price * 2 / 10)
			price_new += int(self.config.max_price * 7 / 10)
			rebuy_price = 0

		price_new = min(9, price_new)
		assert price_refurbished <= price_new, 'The price for used products should be lower or equal to the price of new products'
		return self.convert_price_format(price_refurbished, price_new, rebuy_price)


class RuleBasedCERebuyAgent(RuleBasedCEAgent):
	"""
	This vendor's policy is a version of the RuleBasedCEAgent with rebuy price.
	"""
	def convert_price_format(self, price_refurbished, price_new, rebuy_price):
		return (price_refurbished, price_new, rebuy_price)


class RuleBasedCERebuyAgentCompetitive(RuleBasedAgent, CircularAgent):
	"""
	This vendor's policy is aiming to succeed by undercutting the competitor's prices.
	"""
	def __init__(self, config: HyperparameterConfig, name='rule_based_ce_rebuy_competitive'):
		self.name = name
		self.config = config

	def policy(self, observation, *_) -> tuple:
		assert isinstance(observation, np.ndarray), 'observation must be a np.ndarray'
		# TODO: find a proper way asserting the length of observation (as implemented in AC & QLearning via passing marketplace)

		# in_circulation is ignored
		own_storage = observation[1].item()
		competitors_refurbished_prices, competitors_new_prices, competitors_rebuy_prices = self._get_competitor_prices(observation, True)

		price_new = max(min(competitors_new_prices) - 1, self.config.production_price + 1)
		# competitor's storage is ignored
		if own_storage < self.config.max_storage / 15:
			# fill up the storage immediately
			price_refurbished = min(competitors_refurbished_prices) + 2
			rebuy_price = max(min(competitors_rebuy_prices) + 1, 2)
		elif own_storage < self.config.max_storage / 10:
			# fill up the storage
			price_refurbished = min(competitors_refurbished_prices) + 1
			rebuy_price = min(competitors_rebuy_prices)
		elif own_storage < self.config.max_storage / 8:
			# storage content is ok
			rebuy_price = max(min(competitors_rebuy_prices) - 1, 1)
			price_refurbished = max(min(competitors_refurbished_prices) - 1, rebuy_price + 1)
		else:
			# storage too full, we need to get rid of some refurbished products
			rebuy_price = max(min(competitors_rebuy_prices) - 2, 1)
			price_refurbished = max(round(np.quantile(competitors_refurbished_prices, 0.75)) - 2, rebuy_price + 1)

		return (self._clamp_price(price_refurbished), self._clamp_price(price_new), self._clamp_price(rebuy_price))


class RuleBasedCERebuyAgentStorageMinimizer(RuleBasedAgent, CircularAgent):
	"""
	This vendor's policy reacts to the competitors' prices and minimizes the usage of storage.
	"""
	def __init__(self, config: HyperparameterConfig, name='rule_based_ce_rebuy_stockist'):
		self.name = name
		self.config = config

	def policy(self, observation, *_) -> tuple:
		assert isinstance(observation, np.ndarray), 'observation must be a np.ndarray'
		# TODO: find a proper way asserting the length of observation (as implemented in AC & QLearning via passing marketplace)

		# in_circulation is ignored
		own_storage = observation[1].item()
		competitors_refurbished_prices, competitors_new_prices, competitors_rebuy_prices = self._get_competitor_prices(observation, True)

		price_new = max(median(competitors_new_prices) - 1, self.config.production_price + 1)
		# competitor's storage is ignored
		if own_storage < self.config.max_storage / 15:
			# fill up the storage immediately
			price_refurbished = max(competitors_new_prices + competitors_refurbished_prices)
			rebuy_price = price_new - 1
		else:
			# storage too full, we need to get rid of some refurbished products
			rebuy_price = min(competitors_rebuy_prices) - self.config.max_price / 0.1
			# rebuy_price = min(competitors_rebuy_prices + competitors_new_prices + competitors_refurbished_prices)
			price_refurbished = int(np.quantile(competitors_refurbished_prices, 0.25))

		return (self._clamp_price(price_refurbished), self._clamp_price(price_new), self._clamp_price(rebuy_price))
