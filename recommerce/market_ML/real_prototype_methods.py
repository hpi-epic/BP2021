import numpy as np
import pandas as pd
import torch

N = 200  # Trainingszeilen (historical data)
c_new = 4


def generate_training_data():
	# x{k in 0..24, i in 0..N} default if k=0 then 1 else round(np.random.uniform(0, 20)) # historical/training data
	x_np = torch.tensor([[(1 if k == 0 else np.round(np.random.uniform(0, 20))) for k in range(24 + 1)] for _ in range(N + 1)])
	# historical/training data

	x = x_np.transpose(0, 1)
	# print(x.size())
	# own period: 1st half: own x[7, i]vs x[2, i]| 2nd half: own x[7, i] vs x[2, i+1]
	# comp period: 1st half: com x[2, i-1] vs x[7, i-1] | 2nd half: com x[2, i-1] vs x[7, i]

	for i in range(1, N + 1):
		x[1, i] = x[1, i-1]-x[12, i-1]+x[13, i-1]  # inventory own

		x[2, i] = x[7, i-1]-1 + np.round_(x[5, i-1]/200) if 9 < x[7, i-1] <= 20 else 20  # comp price new
		x[3, i] = x[8, i-1]-1 + np.round_(x[5, i-1]/200) if 5 < x[8, i-1] <= 15 else 15  # comp price used
		x[4, i] = x[9, i-1]-0.5-np.round_(x[5, i-1]/200)if 1 < x[9, i-1] <= 5 else 5  # comp price rebuy
		x[5, i] = x[5, i-1]-x[16, i-1] + x[17, i-1]  # comp inventory
		x[6, i] = max(0, np.round_(0.8*x[6, i-1])+x[11, i-1]+x[12, i-1]-x[13, i-1] + x[15, i-1] + x[16, i-1]-x[17, i-1])  # resource in use

		x[7, i] = np.round_(np.random.uniform(5, 20)) if i < 20 else x[2, i] - 1 + np.round_(x[1, i]/200)if 9 < x[2, i] <= 18 else 18
		# own price new
		x[8, i] = np.round_(np.random.uniform(0, 20)) if i < 20 else x[3, i] - 1 + np.round_(x[1, i]/200)if 5 < x[3, i] <= 12 else 12
		# own price used
		x[9, i] = np.round_(np.random.uniform(0, 10)) if i < 20 else x[4, i]-0.5-np.round_(x[1, i]/200)if 1 < x[4, i] <= 5 else 5
		# own price rebuy
		x[10, i] = x[1, i]*0.05  # own holding cost
		x[22, i] = x[7, i]-1+np.round_(x[5, i]/200) if 9 < x[7, i] <= 20 else 20  # comp new reaction
		x[23, i] = x[8, i]-1+np.round_(x[5, i]/200) if 5 < x[8, i] <= 15 else 15  # comp used reaction
		x[24, i] = x[9, i]-0.5 - np.round_(x[5, i]/200) if 1 < x[9, i] <= 5 else 5  # comp rebuy reaction
		x[11, i] = np.round_(max(0, np.random.uniform(5, 15)-x[7, i] + x[2, i]/4 + x[22, i]/4))  # own sales new
		x[12, i] = np.round_(min(x[1, i],  max(0, np.random.uniform(5, 15)-x[8, i] + x[3, i]/4 + x[23, i]/4)))  # own sales used
		x[13, i] = np.round_(min(x[6, i]/2, max(0, np.random.uniform(5, 15)+x[9, i] - x[4, i]/4 - x[24, i]/4)))  # own repurchases
		x[14, i] = x[5, i]*0.05  # comp holding cost
		x[15, i] = np.round_(max(0, np.random.uniform(5, 15)-x[2, i] + x[7, i]/4 + x[7, i-1]/4))  # comp sales new
		x[16, i] = np.round_(min(x[5, i],  max(0, np.random.uniform(5, 15)-x[3, i] + x[8, i]/4 + x[8, i-1]/4)))  # comp sales used
		x[17, i] = np.round_(min(x[6, i]/2, max(0, np.random.uniform(5, 15)+x[4, i] - x[9, i]/4 - x[9, i-1]/4)))  # comp repurchases

		x[18, i] = -x[10, i] + x[11, i]*(x[7, i]-c_new) + x[12, i]*x[8, i] - x[13, i]*x[9, i]  # own total rewards
		x[19, i] = -x[14, i] + x[15, i]*(x[2, i]-c_new) + x[16, i]*x[3, i] - x[17, i]*x[4, i]  # comp total rewards
		x[20, i] = x[18, i] + x[20, i-1] if i > 0 else 0  # own total accumulated rewards
		x[21, i] = x[19, i] + x[21, i-1] if i > 0 else 0  # comp total accumulated rewards
	return x


def first_regression(data, x_rows, y_index):
	x = data

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


def fourth_regression(data, x_rows, xx_rows, y_index):
	x = data
	y6 = torch.tensor([x[y_index, i+1] for i in range(0, N + 1 - 1)])

	x_y6_x = torch.tensor([[1 if x[9, i] < k else 0 for k in xx_rows] for i in range(1, N + 1)])  # matrix for price own new (the 0 and 1 stuff
	x_y6 = torch.tensor(x[x_rows, : N])
	print('x_y6:', x_y6.size())
	print('x_y6_x:', x_y6_x.size())
	x_y6_combined = torch.concat((x_y6.transpose(0, 1), x_y6_x), 1)

	result_tuple_y6 = torch.linalg.lstsq(x_y6_combined, y6)

	b6_b6x = result_tuple_y6[0]
	print(len(x_rows))
	b6 = b6_b6x[0:len(x_rows)]
	b6x = b6_b6x[len(x_rows):]

	return b6, b6x


def comp_prices(Mi, Mix, bi, bix, flag: str, xb, i):
	xb_first_index = -1
	if flag == 'new':
		xb_first_index = 7
	elif flag == 'used':
		xb_first_index = 8
	elif flag == 'rebuy':
		xb_first_index = 9
	else:
		assert False

	# xb[4,i] = sum{k in M6}  b6[k] *xb[k,i-1] + sum{k in M6x} b6x[k]*(if xb[9,i-1]<k then 1 else 0)  # comp price rebuy (old)
	# xb[24,i]= sum{k in M6}  b6[k] *xb[k,i] + sum{k in M6x} b6x[k]*(if xb[9,i]<k then 1 else 0)  # comp price rebuy (updated)
	return sum([bi[ki] * xb[k, i-1] for ki, k in enumerate(Mi)]) + \
		sum([bix[ki] * (1 if xb[xb_first_index, i] < k else 0) for ki, k in enumerate(Mix)])


def simulation_model_b(data, b1, b2, b3, b4, b4x, b5, b5x, b6, b6x, M1, M2, M3, M4, M5, M6, M4x, M5x, M6x):
	NB = N + 1
	x = data
	# param xb{k in 0..24,i in 0..NB} default if k=0 then 1 else if i=0 then 5 else -1      # simulated data
	xb = torch.tensor([[(1. if k-1 == 0 else 5. if i == 0 else -1.) for k in range(0, 25)] for i in range(0, NB)]).transpose(0, 1)
	print(xb.shape[0])
	for i in range(1, NB):
		xb[1, i] = xb[1, i-1] - xb[12, i-1] + xb[13, i-1]

		xb[2, i] = comp_prices(M4, M4x, b4, b4x, 'new', xb, i-1)  # comp price new 		(old)
		xb[3, i] = comp_prices(M5, M5x, b5, b5x, 'used', xb, i-1)  # comp price used 	(old)
		xb[4, i] = comp_prices(M6, M6x, b6, b6x, 'rebuy', xb, i-1)  # comp price rebuy 	(old)

		xb[5, i] = xb[5, i-1] - xb[16, i-1] + xb[17, i-1]  # comp inventory

		xb[6, i] = max(0, np.round_(0.8*xb[6, i-1]) + xb[11, i-1] + xb[12, i-1] - xb[13, i-1] + xb[15, i-1] + xb[16, i-1] - xb[17, i-1])
		# resources in use

		xb[7, i] = xb[2, i] - 1 + np.round_(x[1, i]/200) if 9 < xb[2, i] <= 18 else 18  # own price new
		xb[8, i] = xb[3, i] - 1 + np.round_(x[1, i]/200) if 5 < xb[3, i] <= 12 else 12  # own price used
		xb[9, i] = xb[4, i] - 0.5 - np.round_(x[1, i]/200) if 1 < xb[4, i] <= 5 else 5    # own price rebuy

		xb[10, i] = xb[1, i] * 0.05  # own holding cost

		xb[22, i] = comp_prices(M4, M4x, b4, b4x, 'new', xb, i)  # comp price new 		(updated)
		xb[23, i] = comp_prices(M5, M5x, b5, b5x, 'used', xb, i)  # comp price used 	(updated)
		xb[24, i] = comp_prices(M6, M6x, b6, b6x, 'rebuy', xb, i)  # comp price rebuy 	(updated)

		# xb[11,i]= np.round_(max(0, np.random.uniform(-5,5) + sum{k in M1} b1[k]*xb[k,i]))

		xb[11, i] = np.round_(max(np.random.uniform(-5, 5) + sum([b1[ki] * xb[k, i] for ki, k in enumerate(M1)]), 0))
		# agent sales new
		xb[12, i] = np.round_(min(x[1, i], max(np.random.uniform(-5, 5) + sum([b2[ki] * xb[k, i] for ki, k in enumerate(M2)]), 0)))
		# agent sales used
		xb[13, i] = np.round_(min(x[6, i] / 2, max(np.random.uniform(-5, 5) + sum([b3[ki] * xb[k, i] for ki, k in enumerate(M3)]), 0)))
		# agent sales rebuy

		xb[14, i] = xb[5, i]*0.05
		# TODO: refactor these into loops or something more beautiful
		xb[15, i] = np.round_(max(0, np.random.uniform(-5, 5) + b1[0] * xb[0, i] + b1[1] * xb[5, i] + b1[2] * xb[7, i-1] +
			b1[3] * xb[8, i-1] + b1[4] * xb[9, i-1] + b1[5] * xb[2, i] + b1[6] * xb[3, i] + b1[7] * xb[4, i] + b1[8] * xb[7, i] +
			b1[9] * xb[8, i] + b1[10] * xb[9, i]))  # cf M1 # competitor sales new

		xb[16, i] = np.round_(min(x[5, i],  max(0, np.random.uniform(-5, 5) + b2[0] * xb[0, i] + b2[1] * xb[5, i] + b2[2] * xb[7, i-1] +
			b2[3] * xb[8, i-1] + b2[4] * xb[9, i-1] + b2[5] * xb[2, i] + b2[6] * xb[3, i] + b2[7] * xb[4, i] + b2[8] * xb[7, i] +
			b2[9] * xb[8, i] + b2[10] * xb[9, i])))  # cf M2 # competitor sales used

		xb[17, i] = np.round_(min(x[6, i] / 2, max(0, np.random.uniform(-5, 5) + b3[0] * xb[0, i] + b3[1] * xb[5, i] + b3[2] * xb[7, i-1]
			+ b3[3] * xb[8, i-1] + b3[4] * xb[9, i-1] + b3[5] * xb[2, i] + b3[6] * xb[3, i] + b3[7] * xb[4, i] + b3[8] * xb[7, i]
			+ b3[9] * xb[8, i] + b3[10] * xb[9, i])))  # cf M3 # competitor sales rebuy

		# rewards
		xb[18, i] = -xb[10, i] + xb[11, i] * (xb[7, i] - c_new) + xb[12, i] * xb[8, i] - xb[13, i] * xb[9, i]      # own total rewards
		xb[19, i] = -xb[14, i] + xb[15, i] * (xb[2, i] - c_new) + xb[16, i] * xb[3, i] - xb[17, i] * xb[4, i]      # comp total rewards

		# rewards cumulated
		xb[20, i] = xb[18, i] + (xb[20, i-1] if i > 0 else 0)  # own total accumulated rewards
		xb[21, i] = xb[19, i] + (xb[21, i-1] if i > 0 else 0)  # comp total accumulated rewards

	return xb


if __name__ == '__main__':
	data = generate_training_data()
	M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
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
	M6 = (0, 2, 3, 4, 7, 8, 9)
	M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
	y6_index = 4
	by4, bxy4 = fourth_regression(data, M4, M4x, y4_index)
	by5, bxy5 = fourth_regression(data, M5, M5x, y5_index)
	by6, bxy6 = fourth_regression(data, M6, M6x, y6_index)
	print('by1:', by1)
	print('by2:', by2)
	print('by3:', by3)
	print('by4:', by4)
	print('bxy4:', bxy4)
	print('by5:', by5)
	print('bxy5:', bxy5)
	print('by6:', by6)
	print('bxy6:', bxy6)
	data_simulated = simulation_model_b(data, by1, by2, by3, by4, bxy4, by5, bxy5, by6, bxy6, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)
	pd.DataFrame(data_simulated.transpose(0, 1)).to_csv('data_simulated.csv', index=False)
