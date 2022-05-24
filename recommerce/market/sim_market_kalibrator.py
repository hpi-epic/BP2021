import pandas as pd
import torch
from sim_market_kalibrated import SimMarketKalibrated
from torch.autograd import Variable

import recommerce.monitoring.exampleprinter as exampleprinter
import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market_ML.predictable_agent import PredictableAgent


class LinearRegressionModel(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(input_size, output_size)  # One in and one out

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

	def kalibrate_market(self, data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index):
		self.N = len(data[0])
		by1 = self.first_regression(data, self.M1, y1_index)  # used sales
		by2 = self.first_regression(data, self.M2, y2_index)  # new sales
		by3 = self.first_regression(data, self.M3, y3_index)  # rebuy sales
		by4, bxy4 = self.fourth_regression(data, self.M4, self.M4x, y4_index, 'used')  # used price comp
		by5, bxy5 = self.fourth_regression(data, self.M5, self.M5x, y5_index, 'new')  # new price comp
		by6, bxy6 = self.fourth_regression(data, self.M6, self.M6x, y6_index, 'rebuy')  # rebuy price comp
		print('b1:', by1)
		print('b2:', by2)
		print('b3:', by3)

		print('b4:', by4)
		print('bx4:', bxy4)

		print('b5:', by5)
		print('bx5:', bxy5)

		print('b6:', by6)
		print('bx6:', bxy6)
		# exit()
		return SimMarketKalibrated(self.config, by1, by2, by3, by4, bxy4, by5, bxy5, by6, bxy6,
			self.M1, self.M2, self.M3, self.M4, self.M5, self.M6, self.M4x, self.M5x, self.M6x)

	def first_regression(self, data, x_rows, y_index: int):
		x = data
		# param y3 {i in 1..N} := x[13,i];
		# set M3 := {0,1,2,3,4,7,8,9,22,23,24};
		y3 = torch.tensor([x[i, y_index] for i in range(self.N)])
		# torch.transpose(y3, 0, 1)
		x_y3 = torch.tensor(x[x_rows, 0:self.N])

		transposed_x_y3 = x_y3.transpose(0, 1)
		print()
		print('transposed_x_y3', transposed_x_y3.size())
		print('y3', y3.size())
		print()
		# minimize OLSy3: sum{i in 1..N} ( sum{k in M3} beta3[k]*x[k,i] - y3[i] )^2;
		# objective OLSy3; solve; for{k in M3} let b3[k]:=beta3[k];;
		result_tuple_y3 = torch.linalg.lstsq(transposed_x_y3, y3, driver='gelsd')

		b3 = result_tuple_y3[0]
		assert len(b3) == len(x_rows) and len(b3) == 10, 'len(b3) = ' + str(len(b3))
		# self.new_regression(transposed_x_y3, y3)
		return b3

	def fourth_regression(self, data, x_rows, xx_rows, y_index: int, flag: str):
		# self.fourth_regression(data, self.M4, self.M4x, y4_index, 'used')
		xb_first_index = -1
		if flag == 'used':
			xb_first_index = 22
		elif flag == 'new':
			xb_first_index = 23
		elif flag == 'rebuy':
			xb_first_index = 24
		else:
			assert False
		x = data
		y = torch.tensor([x[i, y_index] for i in range(1, self.N)])
		# flag determines which prices are set
		x_y_x = torch.tensor([[1 if x[i, xb_first_index] < k else 0 for k in xx_rows] for i in range(1, self.N)])
		# matrix for price agent (the 0 and 1 stuff)
		x_y = torch.tensor(x[x_rows, 1: self.N])

		print('y: ', y.size())
		print('x_y:', x_y.size())
		print('x_y_x:', x_y_x.size())
		x_y_combined = torch.concat((x_y.transpose(0, 1), x_y_x), 1)
		print(x_y_x[19])
		# assert y_index == 7, y_index
		print(x[19, y_index])

		result_tuple_y = torch.linalg.lstsq(x_y_combined, y, driver='gelsd')

		b6_b6x = result_tuple_y[0]
		print(len(x_rows))
		b6 = b6_b6x[0:len(x_rows)]
		b6x = b6_b6x[len(x_rows):]

		return b6, b6x

	def new_regression(self, x_tensor: torch.tensor, y_tensor: torch.tensor):
		x_data = Variable(x_tensor)
		y_data = Variable(y_tensor)
		# our model
		our_model = LinearRegressionModel(len(x_tensor), len(y_tensor))

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

		new_var = Variable(torch.Tensor([x_tensor[0]]))
		pred_y = our_model(new_var)
		print('predict (after training)', 4, our_model(new_var).item())
		exit()


def stable_baselines_agent_kalibrator():
	# training_scenario.train_to_calibrate_marketplace()
	data_path = f'{PathManager.data_path}/kalibration_data/training_data_1000.csv'
	print(data_path)
	data_frame = pd.read_csv(data_path)
	data = torch.tensor(data_frame.values).transpose(0, 1)
	M123 = (1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
	y1_index = 11
	y2_index = 12
	y3_index = 13
	M4 = (2, 3, 4, 7, 8, 9)
	M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y4_index = 2
	M5 = (2, 3, 4, 7, 8, 9)
	M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y5_index = 3
	M6 = (2, 3, 4, 7, 8, 9)
	M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	y6_index = 4

	kalibrator = SimMarketKalibrator(M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)
	kalibrated_market = kalibrator.kalibrate_market(data, y1_index, y2_index, y3_index, y4_index, y5_index, y6_index)
	# training_scenario.train_with_calibrated_marketplace(kalibrated_market)
	agent = sbmodel.StableBaselinesSAC(
		marketplace=kalibrated_market,
		load_path='results/trainedModels/Stable_Baselines_SAC_May08_15-02-28/Stable_Baselines_SAC_01999')
	exampleprinter.main_kalibrated_marketplace(kalibrated_market, agent)


if __name__ == '__main__':
	# training_scenario.train_to_calibrate_marketplace()
	data_path = f'{PathManager.data_path}/kalibration_data/training_data_predictable.csv'
	print('Loading data from:', data_path)
	data_frame = pd.read_csv(data_path)
	data = torch.tensor(data_frame.values).transpose(0, 1)
	M123 = (1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
	y1_index = 11  # sales used agent
	y2_index = 12  # sales new agent
	y3_index = 13  # sales rebuy agent #TODO:improve the data by adding  competitor sales as well, now competitor sells just like the agent
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
	# training_scenario.train_with_calibrated_marketplace(kalibrated_market)
	agent = PredictableAgent(config_hyperparameter)
	exampleprinter.main_kalibrated_marketplace(kalibrated_market, agent, config_hyperparameter)
