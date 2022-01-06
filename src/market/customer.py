#!/usr/bin/env python3

import math
from abc import ABC, abstractmethod

import numpy as np

import configuration.utils_sim_market as ut

# import random


class Customer(ABC):
	@abstractmethod
	def generate_purchase_probabilities_from_offer(self, offers, offer_length_per_vendor) -> np.array:
		"""This method receives a list of offers from the market and returns probabilities for all possible purchase decisions.

		Args:
			offers (np.array): The list of offers which contains both the market state and information about the actions of all vendors
			offer_length_per_vendor (int): The number of fields each vendor has in that offer (actions and vendor_specific_state).
			Until now, it is assumed that the common state uses less fields than one vendor

		Returns:
			np.array: probability distribution for all possible purchase decisions.
			In the first field, there is the probability that the customer does not buy anything.
			In the subsequent fields, there are the probabilites for buying the specific offers from the vendor.
			Look subclass implementation for more details.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')


class CustomerLinear(Customer):
	def generate_purchase_probabilities_from_offer(self, offers, offer_length_per_vendor) -> np.array:
		"""This method receives a list of offers for a linear economy and returns the purchase probability for all vendors.

		Args:
			offers (np.array): Each vendor comes along with two fiels in the array.
			The first field contains the quality and the second field contains the price.
			offer_length_per_vendor (int): The number must be 2 according to the description for offers

		Returns:
			np.array: The first entry contains the probability that a customer does not buy anything.
			Afterwards, the probabilities for all vendors follow.
		"""
		assert offer_length_per_vendor == 2, 'offer_length_per_vendor must be two: one field for the price and one for the quality!'
		nothingpreference = 1
		ratios = [nothingpreference]
		for offer in range(int(len(offers) / 2)):
			quality = offers[2 * offer + 1]
			price = offers[2 * offer] + 1
			ratio = quality / price
			ratios.append(ratio)
		return ut.softmax(np.array(ratios))


class CustomerCircular(Customer):
	def generate_purchase_probabilities_from_offer(self, offers, offer_length_per_vendor) -> np.array:
		"""This method receives a list of offers for a circular economy and returns the purchase probability for the refurbished and new product of all vendors.
		It is assumed that all vendors do have the same quality and same reputation.
		The customer values a second-hand-product 55% compared to a new one.

		Args:
			offers (np.ndarray): First, the array contains the number of products in circulation (ignored in this model).
			Afterwards, the price of refurbished product, new product, potentially rebuy price (ignored) and the storage (ignored) follow.
			offer_length_per_vendor (int): The number of fields each vendor has in that offer (actions and vendor_specific_state).
			Until now, it is assumed that the common state uses less fields than one vendor.
			In the circular economy, the value must be 3 or 4.

		Returns:
			np.array: The first entry contains the probability that a customer does not buy anything.
			Afterwards, for each vendor the probabilities for the second-hand and the new product follow.
		"""
		assert isinstance(offers, np.ndarray), 'offers must be a np.ndarray'
		assert len(offers) % offer_length_per_vendor == 1, 'there must be exactly one field for common state (in_circulation)'
		assert offer_length_per_vendor == 3 or offer_length_per_vendor == 4, 'offer_length_per_vendor needs to be 3 or 4'
		nothingpreference = 1
		preferences = [nothingpreference]
		for offer in range(int(np.floor(len(offers) / offer_length_per_vendor))):
			price_refurbished = offers[1 + offer * offer_length_per_vendor] + 1
			price_new = offers[2 + offer * offer_length_per_vendor] + 1
			assert price_refurbished >= 1 and price_new >= 1, 'price_refurbished and price_new need to be >= 1'

			ratio_old = 5.5 / price_refurbished - math.exp(price_refurbished - 5)
			ratio_new = 10 / price_new - math.exp(price_new - 8)
			preferences += [ratio_old, ratio_new]

		return ut.softmax(np.array(preferences))
