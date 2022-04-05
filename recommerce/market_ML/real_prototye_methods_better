import numpy as np
import torch

N = 200  # Trainingszeilen (historical data)
c_new = 4


def generate_training_data():
	# x{k in 0..24, i in 0..N} default if k=0 then 1 else round(np.random.uniform(0, 20)) # historical/training data
	x = np.array([[(1 if k == 0 else np.round(np.random.uniform(0, 20))) for k in range(N)] for _ in range(24)])  # historical/training data

	# own period: 1st half: own x[6, i]vs x[1, i]| 2nd half: own x[6, i] vs x[1, i+1]
	# comp period: 1st half: com x[1, i-1] vs x[6, i-1] | 2nd half: com x[1, i-1] vs x[6, i]

	for i in range(1, N):
		x[0, i] = x[0, i-1]-x[11, i-1]+x[12, i-1]  # inventory own

		x[1, i] = x[6, i-1]-1 + np.round_(x[4, i-1]/200) if 9 < x[6, i-1] <= 20 else 20  # comp price new
		x[2, i] = x[7, i-1]-1 + np.round_(x[4, i-1]/200) if 5 < x[7, i-1] <= 15 else 15  # comp price used
		x[3, i] = x[8, i-1]-0.5-np.round_(x[4, i-1]/200)if 1 < x[8, i-1] <= 5 else 5  # comp price rebuy
		x[4, i] = x[4, i-1]-x[15, i-1] + x[16, i - 1]  # comp inventory
		x[5, i] = max(0, np.round_(0.8*x[5, i-1])+x[10, i-1]+x[11, i-1]-x[12, i-1] + x[14, i-1] + x[15, i-1]-x[16, i-1])  # resource in use

		x[6, i] = np.round_(np.random.uniform(5, 20)) if i < 20 else x[1, i] - 1 + np.round_(x[0, i]/200)if 9 < x[1, i] <= 18 else 18
		# own price new
		x[7, i] = np.round_(np.random.uniform(0, 20)) if i < 20 else x[2, i] - 1 + np.round_(x[0, i]/200)if 5 < x[2, i] <= 12 else 12
		# own price used
		x[8, i] = np.round_(np.random.uniform(0, 10)) if i < 20 else x[3, i]-0.5-np.round_(x[0, i]/200)if 1 < x[3, i] <= 5 else 5
		# own price rebuy
		x[9, i] = x[0, i]*0.05  # own holding cost
		x[21, i] = x[6, i]-1+np.round_(x[4, i]/200) if 9 < x[6, i] <= 20 else 20  # comp new reaction
		x[22, i] = x[7, i]-1+np.round_(x[4, i]/200) if 5 < x[7, i] <= 15 else 15  # comp used reaction
		x[23, i] = x[8, i]-0.5 - np.round_(x[4, i]/200) if 1 < x[8, i] <= 5 else 5  # comp rebuy reaction
		x[10, i] = np.round_(max(0, np.random.uniform(5, 15)-x[6, i] + x[1, i]/4 + x[21, i]/4))  # own sales new
		x[11, i] = np.round_(min(x[0, i],  max(0, np.random.uniform(5, 15)-x[7, i] + x[2, i]/4 + x[22, i]/4)))  # own sales used
		x[12, i] = np.round_(min(x[5, i]/2, max(0, np.random.uniform(5, 15)+x[8, i] - x[3, i]/4 - x[23, i]/4)))  # own repurchases
		x[13, i] = x[4, i]*0.05  # comp holding cost
		x[14, i] = np.round_(max(0, np.random.uniform(5, 15)-x[1, i] + x[6, i]/4 + x[6, i-1]/4))  # comp sales new
		x[15, i] = np.round_(min(x[4, i],  max(0, np.random.uniform(5, 15)-x[2, i] + x[7, i]/4 + x[7, i-1]/4)))  # comp sales used
		x[16, i] = np.round_(min(x[5, i]/2, max(0, np.random.uniform(5, 15)+x[3, i] - x[8, i]/4 - x[8, i-1]/4)))  # comp repurchases

		x[17, i] = -x[9, i] + x[10, i]*(x[6, i]-c_new) + x[11, i]*x[7, i] - x[12, i]*x[8, i]  # own total rewards
		x[18, i] = -x[13, i] + x[14, i]*(x[1, i]-c_new) + x[15, i]*x[2, i] - x[16, i]*x[3, i]  # comp total rewards
		x[19, i] = x[17, i] + x[19, i-1] if i > 0 else 0  # own total accumulated rewards
		x[20, i] = x[18, i] + x[20, i-1] if i > 0 else 0  # comp total accumulated rewards
	return x


def first_regression(data, x_rows, y_index):
	x = data
	y3 = [x[y_index, i] for i in range(N)]
	x_y3 = x[x_rows, :]

	# param y3 {i in 1..N} := x[12,i];
	# set M3 := {0,1,2,3,4,7,8,9,22,23,24};
	# param b3 {k in M3} default 0;
	# var beta3 {k in M3};

	transposed_tensor_x_y3 = torch.tensor(x_y3).transpose(0, 1)
	tensor_y3 = torch.tensor(y3)

	print('x_y4:', torch.tensor(x_y3).size())
	print('y3:', tensor_y3.size())
	print('transposed_tensor_x_y3:', transposed_tensor_x_y3.size())

	result_tuple_y3 = torch.linalg.lstsq(transposed_tensor_x_y3, tensor_y3)
	print('result:', result_tuple_y3)
	b3 = result_tuple_y3[0]
	# minimize OLSy3: sum{i in 1..N} ( sum{k in M3} beta3[k]*x[k,i] - y3[i] )^2;
	# objective OLSy3; solve; for{k in M3} let b3[k]:=beta3[k];;
	return b3


def fourth_regression(data, x_rows, xx_rows, y_index):
	x = data
	y6 = [x[y_index, i+1] for i in range(0, N-1)]

	x_y6_x = [[1 if x[8, i] < k else 0 for k in xx_rows] for i in range(1, N)]  # matrix for price own new (the 0 and 1 stuff)

	x_y6 = x[x_rows, :(len(x[0]) - 1)]

	x_y6_tensor = torch.tensor(x_y6)
	x_y6_x_tensor = torch.tensor(x_y6_x)
	tensor_y6 = torch.tensor(y6)
	x_y6_combined = torch.concat((x_y6_tensor, x_y6_x_tensor.transpose(0, 1)), 0)

	print('x_y6:', x_y6_tensor.size())
	print('x_y6_x:', x_y6_x_tensor.size())
	print('y6:', tensor_y6.size())
	print('x_y6_combined:', x_y6_combined.size())

	result_tuple_y6 = torch.linalg.lstsq(x_y6_combined.transpose(0, 1), tensor_y6)
	print('result:', result_tuple_y6)
	b6_b6x = result_tuple_y6[0]

	b6 = b6_b6x[0:len(x_rows)]
	b6x = b6_b6x[len(x_rows):]
	print('b6:', b6)
	print('b6x:', b6x)
	return b6, b6x


def simulation_model_b(data, b1, b2, b3, b4, b4x, b5, b5x, b6, b6x, M1, M2, M3):
	NB = N
	x = data
	# param xb{k in 0..24,i in 0..NB} default if k=0 then 1 else if i=0 then 5 else -1      # simulated data
	xb = np.array([[(1 if k-1 == 0 else 5 if i == 0 else -1) for k in range(0, 25)] for i in range(0, NB)]).transpose()
	print(xb.shape[0])
	for i in range(1, NB):
		xb[1, i] = xb[1, i-1]-xb[12, i-1]+xb[13, i-1]

		# xb[2,i] = sum{k in M4}  b4[k] *xb[k,i-1] + sum{k in M4x} (b4x[k]*(1 if xb[7,i-1]<k else 0)) for k in M4x
		# xb[2,i] = [(b4[k] *xb[k,i-1] + (b4x[k2]*(1 if xb[7,i-1]<k2 else 0)) for k2 in M4x) for k in M4]  # comp price new
		for ki, k in enumerate(M4):
			xb[2, i] += b4[ki] * xb[ki, k-1]
		for ki, k in enumerate(M4x):
			xb[2, i] += (b4x[ki] * (1 if xb[7, i-1] < k else 0))  # comp price new

		# xb[3,i] = sum{k in M5}  b5[k] *xb[k,i-1] + sum{k in M5x} b5x[k]*(if xb[8,i-1]<k then 1 else 0)  # comp price used
		for ki, k in enumerate(M5):
			xb[3, i] += b5[ki] * xb[ki, k-1]
		for ki, k in enumerate(M5x):
			xb[3, i] += b5x[ki]*(1 if xb[8, i-1] < k else 0)  # comp price used

		xb[4, i] = sum([b6[ki] * xb[ki, k-1] for ki, k in enumerate(M6)]) \
			+ sum([b6x[ki]*(1 if xb[9, i-1] < k else 0) for ki, k in enumerate(M6x)])
		# xb[4,i] = sum{k in M6}  b6[k] *xb[k,i-1] + sum{k in M6x} b6x[k]*(if xb[9,i-1]<k then 1 else 0)  # comp price rebuy

		xb[5, i] = xb[5, i-1] - xb[16, i-1] + xb[17, i-1]  # comp inventory

		xb[6, i] = max(0, np.round_(0.8*xb[6, i-1]) + xb[11, i-1] + xb[12, i-1] - xb[13, i-1] + xb[15, i-1] + xb[16, i-1] - xb[17, i-1])
		# resources in use

		xb[7, i] = xb[2, i] - 1 + np.round_(x[0, i]/200) if 9 < xb[2, i] <= 18 else 18  # own price new
		xb[8, i] = xb[3, i] - 1 + np.round_(x[0, i]/200) if 5 < xb[3, i] <= 12 else 12  # own price used
		xb[9, i] = xb[4, i] - 0.5 - np.round_(x[0, i]/200) if 1 < xb[4, i] <= 5 else 5  # own price rebuy
		xb[10, i] = xb[1, i] * 0.05  # own holding cost

		# xb[22,i]= sum{k in M4}  b4[k] *xb[k,i] + sum{k in M4x} b4x[k]*(if xb[7,i]<k then 1 else 0)  # comp new reaction
		for ki, k in enumerate(M4):
			xb[22, i] += b4[ki] * xb[ki, i]
		for ki, k in enumerate(M4x):
			b4x[ki] * (1 if xb[7, i] < k else 0)

		# xb[23,i]= sum{k in M5}  b5[k] *xb[k,i] + sum{k in M5x} b5x[k]*(if xb[8,i]<k then 1 else 0)  # comp used reaction
		for ki, k in enumerate(M5):
			xb[23, i] += b5[ki] * xb[k, i]
		for ki, k in enumerate(M5x):
			xb[23, i] += b5x[ki] * (1 if xb[8, i] < k else 0)
		# xb[24,i]= sum{k in M6}  b6[k] *xb[k,i] + sum{k in M6x} b6x[k]*(if xb[9,i]<k then 1 else 0)  # comp rebuy reaction
		for ki, k in enumerate(M6):
			xb[24, i] += b6[ki] * xb[k, i]
		for ki, k in enumerate(M6x):
			xb[24, i] += b6x[ki]*(1 if xb[9, i] < k else 0)

		# xb[11,i]= np.round_(max(0, np.random.uniform(-5,5) + sum{k in M1} b1[k]*xb[k,i]))
		xb[11, i] = np.round_(max(np.random.uniform(-5, 5) + sum([b1[ki] * xb[k, i] for ki, k in enumerate(M1)]), 0))

		xb[12, i] = np.round_(min(x[0, i], max(np.random.uniform(-5, 5) + sum([b2[ki] * xb[k, i] for ki, k in enumerate(M2)]), 0)))
		xb[13, i] = np.round_(min(x[5, i] / 2, max(np.random.uniform(-5, 5) + sum([b3[ki] * xb[k, i] for ki, k in enumerate(M3)]), 0)))
		xb[14, i] = xb[5, i]*0.05
		# M1: 0-0;1-1;2-2;3-3;4-4;7-5;8-6;9-7;22-8;23-9;24-10
		xb[15, i] = np.round_(max(0, np.random.uniform(-5, 5) + b1[0] * xb[0, i] + b1[1] * xb[5, i] + b1[2] * xb[7, i-1] + b1[3] * xb[8, i-1]
			+ b1[4] * xb[9, i-1] + b1[5] * xb[2, i] + b1[6] * xb[3, i] + b1[7] * xb[4, i] + b1[8] * xb[7, i] + b1[9] * xb[8, i]
			+ b1[10] * xb[9, i]))  # cf M1

		xb[16, i] = np.round_(min(x[4, i],  max(0, np.random.uniform(-5, 5) + b2[0] * xb[0, i] + b2[1] * xb[5, i] + b2[2] * xb[7, i-1]
			+ b2[3] * xb[8, i - 1]
			+ b2[4] * xb[9, i-1] + b2[5] * xb[2, i] + b2[6] * xb[3, i] + b2[7] * xb[4, i] + b2[8] * xb[7, i] + b2[9] * xb[8, i]
			+ b2[10] * xb[9, i])))  # cf M2

		xb[17, i] = np.round_(min(x[5, i]/2, max(0, np.random.uniform(-5, 5) + b3[0] * xb[0, i] + b3[1] * xb[5, i] + b3[2] * xb[7, i-1]
			+ b3[3] * xb[8, i-1] + b3[4] * xb[9, i-1] + b3[5] * xb[2, i] + b3[6] * xb[3, i] + b3[7] * xb[4, i] + b3[8] * xb[7, i]
			+ b3[9] * xb[8, i] + b3[10] * xb[9, i])))  # cf M3

		xb[18, i] = -xb[10, i]+xb[11, i]*(xb[7, i] - c_new)+xb[12, i]*xb[8, i]-xb[13, i]*xb[9, i]      # own total rewards
		xb[19, i] = -xb[14, i]+xb[15, i]*(xb[2, i] - c_new)+xb[16, i]*xb[3, i]-xb[17, i]*xb[4, i]      # comp total rewards
		xb[20, i] = xb[18, i] + (xb[20, i-1] if i > 0 else 0)  # own total accumulated rewards
		xb[21, i] = xb[19, i] + (xb[21, i-1] if i > 0 else 0)  # comp total accumulated rewards
	return xb


if __name__ == '__main__':
	data = generate_training_data()
	M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23)
	y1_index = 11
	y2_index = 12
	y3_index = 13
	by1 = first_regression(data, M123, y1_index)
	by2 = first_regression(data, M123, y2_index)
	by3 = first_regression(data, M123, y3_index)

	M4 = (0, 2, 3, 4, 7, 8, 9)
	M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y4_index = 2
	M5 = (0, 2, 3, 4, 7, 8, 9)
	M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
	y5_index = 3
	M6 = (0, 1, 2, 3, 4, 7, 8)
	M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	y6_index = 4
	b4y, b4xy = fourth_regression(data, M4, M4x, y4_index)
	b5y, b5xy = fourth_regression(data, M5, M5x, y5_index)
	b6y, b6xy = fourth_regression(data, M6, M6x, y6_index)
	data_simulated = simulation_model_b(data, by1, by2, by3, b4y, b4xy, b5y, b5xy, b6y, b6xy, M123, M123, M123)
