import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import recommerce.configuration.utils as ut
from recommerce.configuration.path_manager import PathManager
from recommerce.market.customer import Customer


class CustomerCircular(Customer):
	def generate_purchase_probabilities_from_offer(self, market_config, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This method calculates the purchase probability for each vendor in a linear setup.
		It is assumed that all vendors do have the same quality and same reputation.
		The customer values a second-hand-product 55% compared to a new one.

		Check the docstring in the superclass for interface description.
		"""
		assert isinstance(common_state, np.ndarray), 'common_state must be a np.ndarray'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		nothingpreference = 1
		preferences = [nothingpreference]
		for vendor_idx in range(len(vendor_actions)):
			price_refurbished = vendor_actions[vendor_idx][0] + 1
			price_new = vendor_actions[vendor_idx][1] + 1
			assert price_refurbished >= 1 and price_new >= 1, 'price_refurbished and price_new need to be >= 1'

			ratio_old = market_config.compared_value_old * 10 / price_refurbished - np.exp(price_refurbished - market_config.upper_tolerance_old)
			ratio_new = 10 / price_new - np.exp(price_new - market_config.upper_tolerance_new)
			preferences += [ratio_old, ratio_new]

		return ut.softmax(np.array(preferences))


class LinearRegressionCustomer(Customer):
	def create_x_with_binary_features(self, X):
		X_dash_list = []
		for price_threshhold in range(10):
			# iterate throw the columns
			for i_feature, column in enumerate(X.T):
				column_values = np.where(column > price_threshhold, 1, 0)
				# append the new column to X
				X_dash_list.append(column_values.reshape(-1, 1))
		X_dash = np.concatenate(X_dash_list, axis=1)
		return np.concatenate((X, X_dash), axis=1)

	def __init__(self) -> None:
		if not hasattr(LinearRegressionCustomer, 'regressor'):
			customers_dataframe = pd.read_excel(os.path.join(PathManager.results_path, 'customers_dataframe.xlsx'))
			print('Dataset read')
			X = customers_dataframe.iloc[:, 0:6].values
			# X = self.create_x_with_binary_features(X)
			Y = customers_dataframe.iloc[:, 6:9].values

			LinearRegressionCustomer.regressor = LinearRegression()
			LinearRegressionCustomer.regressor.fit(X, Y)
			print(f'LinearRegressionCustomer: R^2 = {self.regressor.score(X, Y)}')

	def generate_purchase_probabilities_from_offer(self, market_config, common_state, vendor_specific_state, vendor_actions) -> np.array:
		assert isinstance(common_state, np.ndarray), 'common_state must be a np.ndarray'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		input_array_customer = np.array(list(vendor_actions[0]) + list(vendor_actions[1])).reshape(1, -1)
		# input_array_customer = self.create_x_with_binary_features(input_array_customer)
		prediction_for_customer = LinearRegressionCustomer.regressor.predict(input_array_customer)[0]

		input_array_competitor = np.array(list(vendor_actions[1]) + list(vendor_actions[0])).reshape(1, -1)
		# input_array_competitor = self.create_x_with_binary_features(input_array_competitor)
		prediction_for_competitor = LinearRegressionCustomer.regressor.predict(input_array_competitor)[0]

		prediction = np.concatenate((prediction_for_customer, prediction_for_competitor[1:3]))

		prediction = np.where(prediction < 0, 0, prediction)

		return prediction


if __name__ == '__main__':
	LinearRegressionCustomer()
