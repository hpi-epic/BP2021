import numpy as np
import torch
from sim_market_kalibrated import SimMarketKalibrated


class SimMarketKalibrator:

	def __init__(self, M1, M2, M3, M4, M5, M6, M4x, M5x, M6x):
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
		by1 = self.first_regression(data, self.M1, y1_index)
		by2 = self.first_regression(data, self.M2, y2_index)
		by3 = self.first_regression(data, self.M3, y3_index)
		by4, bxy4 = self.fourth_regression(data, self.M4, self.M4x, y4_index, 'new')
		by5, bxy5 = self.fourth_regression(data, self.M5, self.M5x, y5_index, 'used')
		by6, bxy6 = self.fourth_regression(data, self.M6, self.M6x, y6_index, 'rebuy')
		return SimMarketKalibrated(by1, by2, by3, by4, bxy4, by5, bxy5, by6, bxy6,
			self.M1, self.M2, self.M3, self.M4, self.M5, self.M6, self.M4x, self.M5x, self.M6x)

	def first_regression(data, x_rows, y_index: int):
		x = data
		N = len(data)
		# param y3 {i in 1..N} := x[13,i];
		# set M3 := {0,1,2,3,4,7,8,9,22,23,24};
		y3 = torch.tensor([x[y_index, i] for i in range(N + 1)])
		x_y3 = torch.tensor(x[x_rows, :])

		transposed_x_y3 = x_y3.transpose(0, 1)

		# minimize OLSy3: sum{i in 1..N} ( sum{k in M3} beta3[k]*x[k,i] - y3[i] )^2;
		# objective OLSy3; solve; for{k in M3} let b3[k]:=beta3[k];;
		result_tuple_y3 = torch.linalg.lstsq(transposed_x_y3, y3)

		b3 = result_tuple_y3[0]

		return b3

	def fourth_regression(data, x_rows, xx_rows, y_index: int, flag: str):
		xb_first_index = -1
		if flag == 'new':
			xb_first_index = 7
		elif flag == 'used':
			xb_first_index = 8
		elif flag == 'rebuy':
			xb_first_index = 9
		else:
			assert False
		N = len(data)
		x = data
		y = torch.tensor([x[y_index, i + 1] for i in range(0, N + 1 - 1)])
		# flag determines which prices are set
		x_y_x = torch.tensor([[1 if x[xb_first_index, i] < k else 0 for k in xx_rows] for i in range(1, N + 1)])
		# matrix for price agent (the 0 and 1 stuff)

		x_y = torch.tensor(x[x_rows, : N])
		print('x_y:', x_y.size())
		print('x_y_x:', x_y_x.size())
		x_y_combined = torch.concat((x_y.transpose(0, 1), x_y_x), 1)

		result_tuple_y = torch.linalg.lstsq(x_y_combined, y)

		b6_b6x = result_tuple_y[0]
		print(len(x_rows))
		b6 = b6_b6x[0:len(x_rows)]
		b6x = b6_b6x[len(x_rows):]

		return b6, b6x


if __name__ == '__main__':
	data = np.array([])
	M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
	y1_index = 11
	y2_index = 12
	y3_index = 13
	M4 = (0, 2, 3, 4, 7, 8, 9)
	M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y4_index = 2
	M5 = (0, 2, 3, 4, 7, 8, 9)
	M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y5_index = 3
	M6 = (0, 2, 3, 4, 7, 8, 9)
	M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	y6_index = 4

	kalibrator = SimMarketKalibrator(M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)
