import pandas as pd
import torch
from sim_market_kalibrated import SimMarketKalibrated
from torch.autograd import Variable

import recommerce.rl.training_scenario as training_scenario
# import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market_ML.predictable_agent import PredictableAgent


class LinearRegressionModel(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1)  # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred


class SimMarketKalibrator:

	def __init__(self, config, M1, M2, M3, M4, M5, M6, M4x, M5x, M6x):
		self.config = config
		self.M1 = M1
		self.M2 = M2
		self.M3 = M3
		self.M4 = M4
		self.M5 = M5
		self.M6 = M6
		self.M4x = M4x
		self.M5x = M5x
		self.M6x = M6x

	def kalibrate_market(self, data, y1_index, y2_index, y3_index, y12_index, y22_index, y32_index, y4_index, y5_index, y6_index):
		self.N = len(data[0])
		by1 = self.first_regression(data, self.M1, y1_index)  # used sales agent
		by2 = self.first_regression(data, self.M2, y2_index)  # new sales agent
		by3 = self.first_regression(data, self.M3, y3_index)  # rebuy sales agent
		# exit()
		by12 = self.first_regression(data, self.M1, y12_index)  # used sales comp
		by22 = self.first_regression(data, self.M2, y22_index)  # new sales comp
		by32 = self.first_regression(data, self.M3, y32_index)  # rebuy sales comp
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
		return SimMarketKalibrated(self.config, by1, by2, by3, by12, by22, by32, by4, bxy4, by5, bxy5, by6, bxy6,
			self.M1, self.M1, self.M4, self.M5, self.M6, self.M4x, self.M5x, self.M6x, data[23, :])

	def first_regression(self, data, x_rows, y_index: int):
		x = data

		# param y3 {i in 1..N} := x[13,i];
		# set M3 := {0,1,2,3,4,7,8,9,22,23,24};
		# y3a = [x[i, y_index] for i in range(1, self.N)]
		if y_index == 15 or y_index == 16 or y_index == 17:
			return torch.zeros(10)
		assert y_index == 11 or y_index == 12 or y_index == 13
		y3 = torch.tensor([x[y_index, i] for i in range(1, self.N)])

		# torch.transpose(y3, 0, 1)
		x_y3 = torch.tensor(x[x_rows, 1: self.N])
		# print(x_y3)
		transposed_x_y3 = x_y3.transpose(0, 1)

		# minimize OLSy3: sum{i in 1..N} ( sum{k in M3} beta3[k]*x[k,i] - y3[i] )^2;
		# objective OLSy3; solve; for{k in M3} let b3[k]:=beta3[k];;
		result_tuple_y3 = torch.linalg.lstsq(transposed_x_y3, y3, driver='gelsd')
		# pt.training(transposed_x_y3, y3)
		# self.new_regression(transposed_x_y3.float(), y3.float())
		b3 = result_tuple_y3[0]
		print('b1:', b3)
		assert len(b3) == len(x_rows) and len(b3) == 11, 'len(b3) = ' + str(len(b3))
		# self.new_regression(transposed_x_y3, y3)
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
		print('olsy1:', olsy1)
		print('loss per value:', olsy1 / (self.N - 1))

		print()
		return b3

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
		x = data
		y = torch.tensor([x[y_index, i] for i in range(1, self.N)])
		# flag determines which prices are set
		x_y_x = torch.tensor([[1 if x[xb_first_index, i] < k else 0 for k in xx_rows] for i in range(1, self.N)])
		# matrix for price agent (the 0 and 1 stuff)
		x_y = torch.tensor(x[x_rows, 1: self.N]).transpose(0, 1)

		# print('y: ', y.size())
		# print('x_y:', x_y.size())
		# print('x_y_x:', x_y_x.size())
		x_y_combined = torch.concat((x_y, x_y_x), 1)
		# print(x_y_x[19])
		# assert y_index == 7, y_index
		# print(x[y_index, 19])

		result_tuple_y = torch.linalg.lstsq(x_y_combined, y, driver='gelsd')

		b6_b6x = result_tuple_y[0]
		# print(len(x_rows))
		b6 = b6_b6x[0:len(x_rows)]
		b6x = b6_b6x[len(x_rows):]

		# print('b6:', b6)
		# print('b6x:', b6x)

		# olsy4 = 0
		# for i in range(1, self.N):
		# 	y1 = x[y_index, i]
		# 	predicted_y1 = (b6[0] * 1
		# 		+ b6[1] * x[1, i]
		# 		+ b6[2] * x[2, i]
		# 		+ b6[3] * x[3, i]
		# 		+ b6[4] * x[4, i]
		# 		+ b6[5] * x[7, i]
		# 		+ b6[6] * x[8, i]
		# 		+ b6[7] * x[9, i])

		# 	olsy4 += (predicted_y1 - y1) ** 2
		# print('olsy4:', olsy4)
		# print('loss per value:', olsy4 / (self.N - 1))

		# olsy4x = 0
		# for i in range(1, self.N):
		# 	y1 = x[y_index, i]
		# 	predicted_y1 = (b6[0] * 1
		# 		+ b6[1] * x[1, i]
		# 		+ b6[2] * x[2, i]
		# 		+ b6[3] * x[3, i]
		# 		+ b6[4] * x[4, i]
		# 		+ b6[5] * x[7, i]
		# 		+ b6[6] * x[8, i]
		# 		+ b6[7] * x[9, i])
		# 	for ki, k in enumerate(xx_rows):
		# 		if x[ki, i] < k:
		# 			predicted_y1 += b6x[ki]

		# 	olsy4x += (predicted_y1 - y1) ** 2
		# print('olsy4x:', olsy4x)
		# print('loss per value:', olsy4x / (self.N - 1))

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

	def new_regression(self, x_tensor: torch.tensor, y_tensor: torch.tensor):
		x_data = Variable(x_tensor[2])
		y_data = Variable(y_tensor)
		# our model
		our_model = LinearRegressionModel(1, 1)

		criterion = torch.nn.MSELoss(size_average=False)
		optimizer = torch.optim.SGD(our_model.parameters(), lr=0.01)

		for epoch in range(500):

			# Forward pass: Compute predicted y by passing
			# x to the model
			pred_y = our_model(x_data)

			# Compute and print loss
			loss = criterion(pred_y, y_data)

			# Zero gradients, perform a backward pass,
			# and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('epoch {}, loss {}'.format(epoch, loss.item()))

		new_var = Variable(x_tensor[0])
		pred_y = our_model(new_var)
		print('predict (after training)', 4, our_model(new_var).item())


def stable_baselines_agent_kalibrator():
	# training_scenario.train_to_calibrate_marketplace()
	data_path = f'{PathManager.data_path}/kalibration_data/training_data_1000.csv'
	print('Loading data from:', data_path)
	data_frame = pd.read_csv(data_path)
	data = torch.tensor(data_frame.values).transpose(0, 1)
	M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
	y1_index = 11  # sales used agent
	y2_index = 12  # sales new agent
	y3_index = 13  # sales rebuy agent
	# TODO: Improve the data by adding competitor sales as well, now competitor sells just like the agent
	M4 = (1, 2, 3, 4, 7, 8, 9)
	M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y4_index = 22
	M5 = (1, 2, 3, 4, 7, 8, 9)
	M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y5_index = 23
	M6 = (1, 2, 3, 4, 7, 8, 9)
	M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	y6_index = 24
	config_hyperparameter = HyperparameterConfigLoader.load('hyperparameter_config')
	kalibrator = SimMarketKalibrator(config_hyperparameter, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)
	kalibrated_market: SimMarketKalibrated = kalibrator.kalibrate_market(data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index)
	training_scenario.train_with_calibrated_marketplace(kalibrated_market)
	# agent = sbmodel.StableBaselinesSAC(
	# 	config=config_hyperparameter,
	# 	marketplace=kalibrated_market,
	# 	load_path='results/trainedModels/Stable_Baselines_SAC_May08_15-02-28/Stable_Baselines_SAC_01999')
	# exampleprinter.main_kalibrated_marketplace(kalibrated_market, agent, config_hyperparameter)


def predictable_agent_market_kalibrator():
	# training_scenario.train_to_calibrate_marketplace()
	# data_path = 'data/kalibration_data/training_data_predictable_int.csv'
	data_path = 'data/kalibration_data/training_data_predictable_int.csv'
	print('Loading data from:', data_path)
	data_frame = pd.read_csv(data_path)
	data = torch.tensor(data_frame.values).transpose(0, 1)
	M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)  # comp prices old, agent prices, comp prices updated
	y1_index = 11  # sales used agent
	y2_index = 12  # sales new agent
	y3_index = 13  # sales rebuy agent

	y12_index = 15  # sales used comp
	y22_index = 16  # sales new comp
	y32_index = 17  # sales rebuy comp
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
	config_hyperparameter = HyperparameterConfigLoader.load('hyperparameter_config')
	kalibrator = SimMarketKalibrator(config_hyperparameter, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)

	kalibrated_market: SimMarketKalibrated = kalibrator.kalibrate_market(
		data, y1_index, y2_index, y3_index, y12_index, y22_index, y32_index, y4_index, y5_index, y6_index)

	# training_scenario.train_with_calibrated_marketplace(kalibrated_market)
	agent = PredictableAgent(config_hyperparameter)
	state = kalibrated_market.reset()
	for i in range(50):
		action = agent.policy(state)
		# print(action)
		state, reward, is_done, _ = kalibrated_market.step(action)
		print(i, action, state, reward, is_done)
	# exampleprinter.main_kalibrated_marketplace(kalibrated_market, agent, config_hyperparameter)


if __name__ == '__main__':
	predictable_agent_market_kalibrator()
