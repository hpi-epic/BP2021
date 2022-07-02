import math
import random

import pandas as pd
import torch

# import recommerce.monitoring.exampleprinter as exampleprinter
# # import recommerce.rl.training_scenario as training_scenario
# # import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
# from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
# from recommerce.configuration.path_manager import PathManager
# # from recommerce.market.sim_market_kalibrated import SimMarketKalibrated
# from recommerce.market_ML.predictable_agent import PredictableAgent
from recommerce.rl.model import simple_network

# from recommerce.rl.stable_baselines import stable_baselines_model as sbmodel

# from attrdict import AttrDict


class LinearRegressionModel(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1)  # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred


class SimMarketKalibrator:

	def __init__(self, config_market, M1, M2, M3, M4, M5, M6, M4x, M5x, M6x):
		self.config_market = config_market
		self.M1 = M1
		self.M2 = M2
		self.M3 = M3
		self.M4 = M4
		self.M5 = M5
		self.M6 = M6
		self.M4x = M4x
		self.M5x = M5x
		self.M6x = M6x

	def kalibrate_market(self, data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index):
		"""Kalibrates a marketplace with the given data. part of Johanns bachelors thesis

		Args:
			data (torch.tensor): The data to kalibrate with
			y1_index (int): Index of the used customer decision of the agent
			y2_index (int): Index of the new customer decision of the agent
			y3_index (int): Index of the rebuy customer decision of the agent
			y12_index (int): deprecated
			y22_index (int): deprecated
			y32_index (int): deprecated
			y4_index (int): Index of the competitors used price
			y5_index (int): Index of the competitors new price
			y6_index (int): Index of the competitors rebuy price

		Returns:
			SimMarketKalibrated: The kalibrated SimMarket
		"""
		self.N = len(data[0])
		# self.jans_regression_without(data, self.M1, y1_index)
		b1 = self.jans_regression_nn(data, self.M2, y2_index)
		print(b1)
		print()
		print()
		by4, bxy4 = self.fourth_regression(data, self.M4, self.M4x, y4_index, 'used')  # used price comp
		by5, bxy5 = self.fourth_regression(data, self.M5, self.M5x, y5_index, 'new')  # new price comp
		by6, bxy6 = self.fourth_regression(data, self.M6, self.M6x, y6_index, 'rebuy')  # rebuy price comp

		# print('b1:', by1)
		# print('b2:', by2)
		# print('b3:', by3)

		# print('b12:', by12)
		# print('b22:', by22)
		# print('b32:', by32)

		print('b4:', by4)
		print('bx4:', bxy4)

		print('b5:', by5)
		print('bx5:', bxy5)

		print('b6:', by6)
		print('bx6:', bxy6)
		# exit()
		print(data.shape)
		reset_state = data[random.randint(1, self.N), :]
		print('reset_state:', reset_state)
		# return SimMarketKalibrated(self.config_market, b1, b1, b1, b1, b1, b1, by4, bxy4, by5, bxy5, by6, bxy6,
		# 	self.M1, self.M1, self.M4, self.M5, self.M6, self.M4x, self.M5x, self.M6x, reset_state)

	def kalibrate_market_betas(self, data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index):
		"""Kalibrates a marketplace with the given data. part of Johanns bachelors thesis

		Args:
			data (torch.tensor): The data to kalibrate with
			y1_index (int): Index of the used customer decision of the agent
			y2_index (int): Index of the new customer decision of the agent
			y3_index (int): Index of the rebuy customer decision of the agent
			y4_index (int): Index of the competitors used price
			y5_index (int): Index of the competitors new price
			y6_index (int): Index of the competitors rebuy price

		Returns:
			betas for customer and competitor behaviour
		"""
		self.N = len(data[0])
		# self.jans_regression_without(data, self.M1, y1_index)
		b1 = self.jans_regression_nn(data, self.M2, y2_index)

		print()
		print()
		by4, bxy4 = self.fourth_regression(data, self.M4, self.M4x, y4_index, 'used')  # used price comp
		by5, bxy5 = self.fourth_regression(data, self.M5, self.M5x, y5_index, 'new')  # new price comp
		by6, bxy6 = self.fourth_regression(data, self.M6, self.M6x, y6_index, 'rebuy')  # rebuy price comp

		print('b4:', by4)
		print('bx4:', bxy4)

		print('b5:', by5)
		print('bx5:', bxy5)

		print('b6:', by6)
		print('bx6:', bxy6)
		# exit()
		print(data.shape)
		reset_state = data[random.randint(1, self.N), :]
		print('reset_state:', reset_state)
		return b1, b1, b1, by4, bxy4, by5, bxy5, by6, bxy6

	def jans_regression_nn(self, data, x_rows, y_index):
		y_index = (11, 12, 13)
		x = torch.index_select(data, 1, torch.IntTensor(x_rows))
		y = torch.index_select(data, 1, torch.IntTensor(y_index))
		# make x and y float tensors
		x = x.float()
		y = y.float()
		x_test = x[-1000:]
		y_test = y[-1000:]
		x = x[:-1000]
		y = y[:-1000]
		batch_size = 32
		model = simple_network(x.shape[1], y.shape[1])
		lossfunction = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
		episodes = 1

		# use torch utils dataloader to iterate over the data
		data_loader = torch.utils.data.DataLoader(
			dataset=torch.utils.data.TensorDataset(x, y),
			batch_size=batch_size,
			shuffle=True
		)
		for i in range(episodes):
			losses = []
			for x_batch, y_batch in data_loader:
				optimizer.zero_grad()
				output = model(x_batch)
				loss = lossfunction(output, y_batch)
				losses.append(loss.item())
				loss.backward()
				optimizer.step()

			print('episode:', i, 'loss:', torch.mean(torch.Tensor(losses)).item())

		final_ys = model(x_test)
		final_loss = lossfunction(final_ys, y_test)
		print('final_loss:', final_loss.item())
		print('final_loss_sqrt:', torch.sqrt(final_loss).item())
		return model

	def jans_regression_without(self, data, x_rows, y_index):
		"""Regression a la Jan without added features

		Args:
			data (torch.tensor): The data to kalibrate with
			x_rows (Tuple): Rows the x has
			y_index (int): Index of the y
		"""
		y_index = (11, 12, 13)
		x = torch.index_select(data, 1, torch.IntTensor(x_rows))
		y = torch.index_select(data, 1, torch.IntTensor(y_index))
		assert x.shape[0] == y.shape[0]

		parameters, _, _, _ = torch.linalg.lstsq(x, y, driver='gelsd')

		predicted = torch.matmul(x, parameters)
		differences = predicted - y
		print('losses', torch.sqrt(torch.mean(differences * differences, 0)))

	def jans_regression_with(self, data, x_rows, y_index):
		y_index = (11, 12, 13)
		x = torch.index_select(data, 1, torch.IntTensor(x_rows))
		y = torch.index_select(data, 1, torch.IntTensor(y_index))
		assert x.shape[0] == y.shape[0]
		print('x.shape:', x.shape)
		number_datapoints = x.shape[0]
		number_features = x.shape[1] - 1
		new_x = x[:, 1:].view((number_datapoints, 1, -1))
		x_append = torch.matmul(new_x.transpose(1, 2), new_x)
		x_append = x_append.view(number_datapoints, -1)
		wanted_features = list(filter(lambda x: x // number_features <= x % number_features, range(number_features * number_features)))
		x_append = torch.index_select(x_append, 1, torch.IntTensor(wanted_features))
		features = torch.cat([x, x_append], 1)

		parameters, _, _, _ = torch.linalg.lstsq(features, y, driver='gelsd')
		# parameters = torch.linalg.solve(features, y)
		# parameters = torch.matmul(torch.matmul(torch.inverse(torch.matmul(features.transpose(0, 1), features)), features.transpose(0, 1)), y)

		predicted = torch.matmul(features, parameters)
		differences = predicted - y
		print('losses', torch.sqrt(torch.mean(differences * differences, 0)))
		return parameters  # [:, 0], parameters[:, 1], parameters[:, 2]

	def first_regression(self, data, x_rows, y_index: int):
		x = data
		print(x)
		# param y3 {i in 1..N} := x[13,i];
		# set M3 := {0,1,2,3,4,7,8,9,22,23,24};
		# y3a = [x[i, y_index] for i in range(1, self.N)]
		if y_index == 15 or y_index == 16 or y_index == 17:
			# assert False
			return torch.zeros(10)
		assert y_index == 11 or y_index == 12 or y_index == 13

		# define the values to learn
		y3 = torch.tensor([x[y_index, i] for i in range(1, self.N)])

		# define the values to learn from
		x_y3 = torch.tensor(x[x_rows, 1: self.N])

		# add sqared matrix magic
		x3 = torch.transpose(x_y3, 0, 1)
		x4 = []
		for vector in x3:

			condensed = self.get_matrix_streched(vector)

			x4.append(torch.cat([vector, condensed]).view(1, -1))

		x4_tensor = torch.cat(x4)
		print(x4_tensor.shape)

		# minimize OLSy3: sum{i in 1..N} ( sum{k in M3} beta3[k]*x[k,i] - y3[i] )^2;
		# objective OLSy3; solve; for{k in M3} let b3[k]:=beta3[k];;

		# do the regression
		result_tuple_y3 = torch.linalg.lstsq(x4_tensor, y3, driver='gelsd')

		# spilt the beta values
		b3_b3x = result_tuple_y3[0]
		b3 = b3_b3x[0:len(x_rows)]
		b3x = b3_b3x[len(x_rows):]

		print('b:', b3_b3x)
		assert len(b3_b3x) == 111, 'len(b3) = ' + str(len(b3_b3x))

		print('new MSE calculation:')
		sse = 0

		# for vector in torch.transpose(x, 0, 1):
		for vi, vector in enumerate(x3):
			y1 = x[y_index, vi]
			result = sum([b3[ki] * vector[ki] for ki, k in enumerate(x_rows)])

			condensed = self.get_matrix_streched(vector)
			result_mat = torch.matmul(b3x, condensed)
			predicted_y1 = result + result_mat
			sse += ((predicted_y1 - y1) ** 2)
		mse = sse / (self.N - 1)
		print('mse:', mse)
		print('rmse', math.sqrt(mse))

		return b3

	def get_matrix_streched(self, vector):
		vector1 = vector[1:]
		v1 = vector1.view(-1, 1)
		v2 = vector1.view(1, -1)
		mat = torch.matmul(v1, v2)
		condensed = mat.view(-1)
		return condensed

	def fourth_regression(self, data, x_rows, xx_rows, y_index: int, flag: str):
		# self.fourth_regression(data, self.M4, self.M4x, y4_index, 'used')

		xb_first_index = -1
		if flag == 'used':
			xb_first_index = 7
		elif flag == 'new':
			xb_first_index = 8
		elif flag == 'rebuy':
			xb_first_index = 9
		else:
			assert False
		x = data.transpose(0, 1)
		y = torch.tensor([x[y_index, i] for i in range(1, self.N)])
		# flag determines which prices are set
		x_y_x = torch.tensor([[1 if x[xb_first_index, i] < k else 0 for k in xx_rows] for i in range(1, self.N)])
		# matrix for agents price  (the 0 and 1 stuff)
		x_y = torch.tensor(x[x_rows, 1: self.N]).transpose(0, 1)

		# concatinate the unchanges values and the modified prices
		x_y_combined = torch.concat((x_y, x_y_x), 1)

		result_tuple_y = torch.linalg.lstsq(x_y_combined, y, driver='gelsd')

		b6_b6x = result_tuple_y[0]
		# print(len(x_rows))
		b6 = b6_b6x[0:len(x_rows)]
		b6x = b6_b6x[len(x_rows):]

		mse = sum(
			[(
				sum([
					b6[ki] * x[k, i]
					for ki, k in enumerate(x_rows)])
				+ sum([
					b6x[ki] * (1 if x[xb_first_index, i] < k else 0)
					for ki, k in enumerate(xx_rows)])
				- x[y_index, i]) ** 2
			for i in range(1, self.N - 1)])
		print('MSE:', mse)
		print('MSE per value:', mse / (self.N - 1))
		print()

		i_example = 374
		examplevalue = sum([b6[ki] * x[k, i_example] for ki, k in enumerate(x_rows)])
		+ sum([b6x[ki] * (1 if x[xb_first_index, i_example] < k else 0) for ki, k in enumerate(xx_rows)])

		print('examplevalue:', round(float(examplevalue), 2))
		print('actual_value:', float(x[y_index, i_example]))
		print()
		return b6, b6x

	def first_regression_old(self, data, x_rows, y_index: int):
		x = data
		print(x)

		if y_index == 15 or y_index == 16 or y_index == 17:
			# assert False
			return torch.zeros(10)

		assert y_index == 11 or y_index == 12 or y_index == 13

		y3 = torch.tensor([x[y_index, i] for i in range(1, self.N)])

		# torch.transpose(y3, 0, 1)
		x_y3 = torch.tensor(x[x_rows, 1: self.N])

		# x_y31 =
		# print(x_y3)
		transposed_x_y3 = x_y3.transpose(0, 1)

		result_tuple_y3 = torch.linalg.lstsq(transposed_x_y3, y3, driver='gelsd')
		b3 = result_tuple_y3[0]

		assert len(b3) == 11, 'len(b3) = ' + str(len(b3))
		olsy1 = 0
		for i in range(1, self.N):
			y1 = x[y_index, i]
			predicted_y1 = (b3[0] * 1
				+ b3[1] * x[1, i]
				+ b3[2] * x[2, i]
				+ b3[3] * x[3, i]
				+ b3[4] * x[4, i]
				+ b3[5] * x[7, i]
				+ b3[6] * x[8, i]
				+ b3[7] * x[9, i]
				+ b3[8] * x[22, i]
				+ b3[9] * x[23, i]
				+ b3[10] * x[24, i])

			olsy1 += (predicted_y1 - y1) ** 2
		print('old mse:', olsy1 / (self.N - 1))
		print('rmse:', math.sqrt(olsy1 / (self.N - 1)))

		# loss for comp sales and prices (old)
		# for i in range(1, self.N):
		# 	y1 = x[y_index + 4, i]
		# 	predicted_y1 = (b3[0] * 1
		# 		+ b3[1] * x[1, i]
		# 		+ b3[2] * x[2, i]
		# 		+ b3[3] * x[3, i]
		# 		+ b3[4] * x[4, i]
		# 		+ b3[5] * x[7, i]
		# 		+ b3[6] * x[8, i]
		# 		+ b3[7] * x[9, i]
		# 		+ b3[8] * x[22, i]
		# 		+ b3[9] * x[23, i]
		# 		+ b3[10] * x[24, i])
		# 	olsy1 += (predicted_y1 - y1) ** 2
		# print('MSE comp:', olsy1)
		# print('RMSE per value:', math.sqrt(olsy1) / (self.N - 1))
		# print()

		return b3


def stable_baselines_agent_kalibrator():
	pass
	# training_scenario.train_to_calibrate_marketplace()
	# data_path = f'{PathManager.data_path}/kalibration_data/training_data_native_marketplace_exploration.csv'
	# print('Loading data from:', data_path)
	# data_frame = pd.read_csv(data_path)
	# data = torch.tensor(data_frame.values).transpose(0, 1)
	# M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
	# y1_index = 11  # sales used agent
	# y2_index = 12  # sales new agent
	# y3_index = 13  # sales rebuy agent
	# # TODO: Improve the data by adding competitor sales as well, now competitor sells just like the agent
	# M4 = (1, 2, 3, 4, 7, 8, 9)
	# M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	# y4_index = 22
	# M5 = (1, 2, 3, 4, 7, 8, 9)
	# M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	# y5_index = 23
	# M6 = (1, 2, 3, 4, 7, 8, 9)
	# M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	# y6_index = 24
	# config_hyperparameter = HyperparameterConfigLoader.load('hyperparameter_config')
	# kalibrator = SimMarketKalibrator(config_hyperparameter, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)
	# # kalibrated_market: SimMarketKalibrated = kalibrator.kalibrate_market(data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index)
	# # training_scenario.train_with_calibrated_marketplace(kalibrated_market)
	# agent = sbmodel.StableBaselinesSAC(
	# 	config=config_hyperparameter,
	# 	marketplace=kalibrated_market,
	# 	load_path='results/trainedModels/Stable_Baselines_SAC_May08_15-02-28/Stable_Baselines_SAC_01999')
	# exampleprinter.main_kalibrated_marketplace(kalibrated_market, agent, config_hyperparameter)


def predictable_agent_market_kalibrator():
	pass
	# training_scenario.train_to_calibrate_marketplace()
	# data_path = 'data/kalibration_data/training_data_predictable_int.csv'
	# data_path = 'data/kalibration_data/training_data_native_marketplace.csv'
	# print('Loading data from:', data_path)
	# data_frame = pd.read_csv(data_path)
	# data = torch.tensor(data_frame.values).transpose(0, 1)
	# M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)  # comp prices old, agent prices, comp prices updated
	# y1_index = 11  # sales used agent
	# y2_index = 12  # sales new agent
	# y3_index = 13  # sales rebuy agent

	# y12_index = 15  # sales used comp
	# y22_index = 16  # sales new comp
	# y32_index = 17  # sales rebuy comp
	# # TODO: Improve the data by adding competitor sales as well, now competitor sells just like the agent
	# M4 = (0, 1, 2, 3, 4, 7, 8, 9)
	# M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	# y4_index = 2
	# M5 = (0, 1, 2, 3, 4, 7, 8, 9)
	# M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	# y5_index = 3
	# M6 = (0, 1, 2, 3, 4, 7, 8, 9)
	# M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	# y6_index = 4
	# config_hyperparameter = HyperparameterConfigLoader.load('hyperparameter_config')
	# kalibrator = SimMarketKalibrator(config_hyperparameter, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)

	# kalibrated_market: SimMarketKalibrated = kalibrator.kalibrate_market(
	# 	data, y1_index, y2_index, y3_index, y12_index, y22_index, y32_index, y4_index, y5_index, y6_index)

	# training_scenario.train_with_calibrated_marketplace(kalibrated_market)
	# agent = PredictableAgent(config_hyperparameter)
	# state = kalibrated_market.reset()
	# for i in range(50):
	# 	action = agent.policy(state)
	# 	# print(action)
	# 	state, reward, is_done, _ = kalibrated_market.step(action)
	# 	print(i, action, state, reward, is_done)
	# exampleprinter.main_kalibrated_marketplace(kalibrated_market, agent, config_hyperparameter)


def jans_kalibrator():
	# training_scenario.train_to_calibrate_marketplace()
	# data_path = 'data/kalibration_data/training_data_predictable_int.csv'
	data_path = 'data/kalibration_data/training_data_native_marketplace_exploration_after_merge.csv'
	print('Loading data from:', data_path)
	data_frame = pd.read_csv(data_path)
	data = torch.tensor(data_frame.values)[1:, :]
	M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)  # comp prices old, agent prices, comp prices updated
	y1_index = 11  # sales used agent
	y2_index = 12  # sales new agent
	y3_index = 13  # sales rebuy agent
	config_market = ''
	# TODO: Improve the data by adding competitor sales as well, now competitor sells just like the agent
	M4 = (0, 1, 2, 3, 4, 7, 8, 9)
	M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y4_index = 2
	M5 = (0, 1, 2, 3, 4, 7, 8, 9)
	M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y5_index = 3
	M6 = (0, 1, 2, 3, 4, 7, 8, 9)
	M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	y6_index = 4
	# config_market: AttrDict = HyperparameterConfigLoader.load('market_config', SimMarketKalibrated)
	kalibrator = SimMarketKalibrator(config_market, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)

	# kalibrated_market: SimMarketKalibrated =
	kalibrator.kalibrate_market(
		data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index)
	# save_path = f'{PathManager.data_path}data/kalibrated_agents/SAC_Agent_Kalibrated_Marketplace_01.zip'
	# training_scenario.train_with_calibrated_marketplace(kalibrated_market, save_path=save_path)


if __name__ == '__main__':
	jans_kalibrator()
