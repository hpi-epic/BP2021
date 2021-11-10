#!/usr/bin/env python3

# helper
import random
import numpy as np

# own files
import utils as ut

class Competitor():   
	def __init__(self):
    		self.maxprice = ut.MAX_PRICE
		self.maxquality = ut.MAX_QUALITY
		self.quality = ut.shuffle_quality()
		
	def get_initial_price(self, random_start):
		if random_start:
			return max(int(ut.PRODUCTION_PRICE + np.random.normal() * 3 + 3), ut.PRODUCTION_PRICE + 1)
		else:
			return ut.PRODUCTION_PRICE

	def reset(self, random_start):
		self.quality = ut.shuffle_quality()
		return self.get_initial_price(random_start), self.quality
		

	def give_competitors_price(self, state):
		return self.comp_strategy1(state)
		

	def comp_strategy1(self, state):
		# this stratgy is based on a price quality ratio
		agent_price = state[0]
		comp_price = state[2]
		agent_quality = state[1]
		comp_quality = state[3]

		ratio = (agent_quality / agent_price) / \
			(comp_quality / comp_price)
		if random.random() < 0.1:
			return comp_price + random.randint(-1, 1)
		elif comp_price < ut.PRODUCTION_PRICE or (ratio < 0.95 and comp_price < ut.MAX_PRICE - 5):
			return agent_price + 1
		elif agent_price > ut.PRODUCTION_PRICE + 1 and ratio > 1.05 or comp_price > ut.MAX_PRICE - 5:
			return agent_price - 1
		elif agent_price > ut.PRODUCTION_PRICE + 1 and comp_quality > agent_quality:
			return agent_price + 1
		else:
			return comp_price

	def comp_strategy2(self, state):
		# this competitor is based on quality and agents actions
		agent_price = state[0]
		comp_price = state[2]
		agent_quality = state[1]
		comp_quality = state[3]

		new_price = 0

		if comp_quality > agent_quality + 15:
    			# significantly better quality
			new_price =  agent_price + 2
		elif comp_quality > agent_quality:
			# slightly better quality
			new_price = agent_price + 1
		elif comp_quality < agent_quality and comp_quality > agent_quality - 15:    
			# slightly worse quality
			new_price = agent_price - 1
		elif comp_quality < agent_quality:
			# significantly worse quality
			new_price = agent_price - 2
		elif comp_quality == agent_quality:
			# same quality
			new_price = comp_price
		if new_price < ut.PRODUCTION_PRICE:
			new_price = ut.PRODUCTION_PRICE + 1
		elif new_price > self.maxprice:
			new_price = self.maxprice
		return new_price
