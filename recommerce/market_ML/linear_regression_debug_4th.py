import pandas as pd
import torch

data_path = 'data/kalibration_data/training_data_predictable_int.csv'
print('Loading data from:', data_path)
data_frame = pd.read_csv(data_path)
x = torch.tensor(data_frame.values).transpose(0, 1)
N = len(x[0])


print(x[:, 1])

y_index = 22
xb_first_index = 7
x_rows = (0, 2, 3, 4, 7, 8, 9)
xx_values = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

y = torch.tensor([x[y_index, i] for i in range(1, N)])
# flag determines which prices are set
x_y_x = torch.tensor([[(1 if x[xb_first_index, i] < k else 0) for k in xx_values] for i in range(1, N)])
# param x6 {k in M6x,i in 1..N-1} := if x[9,i]<k then 1 else 0;
df = pd.DataFrame(x_y_x)
print('x_based values', x[xb_first_index, 0:5])
print('x_y_x:')
print(df.head())

# matrix for price agent (the 0 and 1 stuff)
x_y = x[x_rows, 1: N].transpose(0, 1)
df2 = pd.DataFrame(x_y)
print('x_y:')
print(df2.head())
print()
print('x shape: ', x_y.shape)
print('xx shape: ', x_y_x.shape)
print('y shape: ', y.shape)


x_y_combined = torch.concat((x_y, x_y_x), 1)
df3 = pd.DataFrame(x_y_combined)
print('x_y_combined:')
print(df3.head())
print()

print('x_y_combined shape: ', x_y_combined.shape)

result_tuple_y = torch.linalg.lstsq(x_y_combined, y)

b6_b6x = result_tuple_y[0]
print('b6_b6x:', b6_b6x)
print()
# print(len(x_rows))
b6 = b6_b6x[0:len(x_rows)]
b6x = b6_b6x[len(x_rows):]

print('b6:', b6)
print('b6x:', b6x)

# olsy4 = 0
# for i in range(1, N):
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
# print('loss per value:', olsy4 / (N - 1))


# sum{i in 1..N-1} (
# 	sum{k in M6}  beta6[k] *x[k,i]
# 	+ sum{k in M6x} beta6x[k]*x6[k,i]
# 	- y6[i]
# )^2;
mse = sum(
	[(
		sum([
			b6[ki] * x[k, i]
			for ki, k in enumerate(x_rows)])
		+ sum([
			b6x[ki] * (1 if x[xb_first_index, i] < k else 0)
			for ki, k in enumerate(xx_values)])
		- x[y_index, i]) ** 2
	for i in range(1, N - 1)])

print()
print('MSE:', mse)
print('MSE per value:', mse / (N - 1))
