import pandas as pd
import torch

data_path = 'data/kalibration_data/training_data_predictable_int.csv'
print('Loading data from:', data_path)
data_frame = pd.read_csv(data_path)
data = torch.tensor(data_frame.values).transpose(0, 1)
N = len(data[0])
x = data
x_rows = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
# param y3 {i in 1..N} := x[13,i];
# set M3 := {0,1,2,3,4,7,8,9,22,23,24};
y_index = 11

y3 = torch.tensor([x[y_index, i] for i in range(1, N)])

# torch.transpose(y3, 0, 1)
x_y3 = torch.tensor(x[x_rows, 1: N])
print(x_y3)
transposed_x_y3 = x_y3.transpose(0, 1)

# minimize OLSy3: sum{i in 1..N} ( sum{k in M3} beta3[k]*x[k,i] - y3[i] )^2;
# objective OLSy3; solve; for{k in M3} let b3[k]:=beta3[k];;
result_tuple_y3 = torch.linalg.lstsq(transposed_x_y3, y3, driver='gelsd')
# pt.training(transposed_x_y3, y3)
# new_regression(transposed_x_y3.float(), y3.float())
b3 = result_tuple_y3[0]
print('b1:', b3)
# new_regression(transposed_x_y3, y3)
olsy1 = 0
for i in range(1, N):
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
	# predicted_y1 = (5.66873 * 1 + 0.1541 * x[7, i])
	olsy1 += (predicted_y1 - y1) ** 2
print('olsy1:', int(olsy1))
print('loss per value:', int(olsy1) / N - 1)
