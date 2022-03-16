from abc import ABC

from configuration.hyperparameter_config import config
from market.circular.circular_customers import CustomerCircular
from market.vendors import Agent, FixedPriceAgent, HumanPlayer, RuleBasedAgent


class CircularAgent(Agent, ABC):
	pass


class HumanPlayerCE(CircularAgent, HumanPlayer):
	def __init__(self, name='YOU - Circular'):
		self.name = name
		print('Welcome to this funny game! Now, you are the one playing the game!')

	def policy(self, observation, *_) -> int:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 1, 'Please enter two numbers seperated by spaces!'
		price_old, price_new = raw_input_string.split(' ')
		return (int(price_old), int(price_new))


class HumanPlayerCERebuy(HumanPlayerCE):
	def policy(self, observation, *_) -> int:
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

	def policy(self, *_) -> int:
		return self.fixed_price


class FixedPriceCERebuyAgent(FixedPriceCEAgent):
	def __init__(self, fixed_price=(3, 6, 2), name='fixed_price_ce_rebuy'):
		assert isinstance(fixed_price, tuple), f'fixed_price must be a tuple: {fixed_price} ({type(fixed_price)})'
		assert len(fixed_price) == 3, f'fixed_price must contain three values: {fixed_price}'
		assert all(isinstance(price, int) for price in fixed_price), f'the prices in fixed_price must be integers: {fixed_price}'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> int:
		return self.fixed_price


class RuleBasedCEAgent(RuleBasedAgent, CircularAgent):
	def __init__(self, name='rule_based_ce'):
		self.name = name

	def return_prices(self, price_old, price_new, rebuy_price):
		return (price_old, price_new)

	def policy(self, observation, epsilon=0) -> int:
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
