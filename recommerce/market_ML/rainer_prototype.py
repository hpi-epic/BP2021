import numpy as np
import pandas as pd

N = 200  # Trainingszeilen (historical data)
c_new = 4

# x{k in 0..24, i in 0..N} default if k=0 then 1 else round(np.random.uniform(0, 20)) # historical/training data
x = np.array([[(1 if k == 0 else np.round_(np.random.uniform(0, 20))) for k in range(N)] for _ in range(25)])  # historical/training data


# own period: 1st half: own x[7, i]vs x[2, i]| 2nd half: own x[7, i] vs x[2, i+1]
# comp period: 1st half: com x[2, i-1] vs x[7, i-1] | 2nd half: com x[2, i-1] vs x[7, i]

for i in range(1, N):
	x[1, i] = x[1, i-1]-x[12, i-1]+x[13, i-1]  # inventory own

	x[2, i] = x[7, i-1]-1 + np.round_(x[5, i-1]/200) if 9 < x[7, i-1] <= 20 else 20  # comp price new
	x[3, i] = x[8, i-1]-1 + np.round_(x[5, i-1]/200) if 5 < x[8, i-1] <= 15 else 15  # comp price used
	x[4, i] = x[9, i-1]-0.5-np.round_(x[5, i-1]/200)if 1 < x[9, i-1] <= 5 else 5  # comp price rebuy
	x[5, i] = x[5, i-1]-x[16, i-1] + x[17, i - 1]  # comp inventory
	x[6, i] = max(0, np.round_(0.8*x[6, i-1])+x[11, i-1]+x[12, i-1]-x[13, i-1] + x[15, i-1] + x[16, i-1]-x[17, i-1])  # resource in use

	x[7, i] = np.round_(np.random.uniform(5, 20)) if i < 20 else x[2, i] - 1 + \
		np.round_(x[1, i]/200)if 9 < x[2, i] <= 18 else 18  # own price new
	x[8, i] = np.round_(np.random.uniform(0, 20)) if i < 20 else x[3, i] - 1 + \
		np.round_(x[1, i]/200) if 5 < x[3, i] <= 12 else 12  # own price used
	x[9, i] = np.round_(np.random.uniform(0, 10)) if i < 20 else x[4, i] - 0.5 \
		- np.round_(x[1, i]/200) if 1 < x[4, i] <= 5 else 5  # own price rebuy
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

print('x:', x)
print('x.size', x.size)

x_df = pd.DataFrame(x)
x_df.to_csv('x_first_prototype.csv')
# for {i in range(1, n):
# x[1, i] = x[1, i-1]-x[12, i-1]+x[13, i-1] # inventory own

# x[2, i] = if 9<x[7, i-1]<=20 then x[7, i-1]-1 +round(x[5, i-1]/200) else 20 # comp price new
# x[3, i] = if 5<x[8, i-1]<=15 then x[8, i-1]-1 +round(x[5, i-1]/200) else 15 # comp price used
# x[4, i] = if 1<x[9, i-1]<= 5 then x[9, i-1]-0.5-round(x[5, i-1]/200) else 5 # comp price rebuy
# x[5, i] = x[5, i-1]-x[16, i-1]+x[17, i-1]# comp inventory
# x[6, i] = max(0, round(0.8*x[6, i-1])+x[11, i-1]+x[12, i-1]-x[13, i-1]+x[15, i-1]+x[16, i-1]-x[17, i-1]) # resource in use
# x[7, i] = if i<20 then round(np.random.uniform(5, 20)) else if 9<x[2, i]<=18 then x[2, i]-1 +round(x[1, i]/200) else 18 # own price new
# x[8, i] = if i<20 then round(np.random.uniform(0, 20)) else if 5<x[3, i]<=12 then x[3, i]-1 +round(x[1, i]/200) else 12 # own price used
# x[9, i] = if i<20 then round(np.random.uniform(0, 10)) else if 1<x[4, i]<= 5 then x[4, i]-0.5-round(x[1, i]/200) else 5 # own price rebuy
# x[10, i]= x[1, i]*0.05# own holding cost
# x[22, i]= if 9<x[7, i]<=20 then x[7, i]-1+round(x[5, i]/200) else 20 # comp new reaction
# x[23, i]= if 5<x[8, i]<=15 then x[8, i]-1+round(x[5, i]/200) else 15 # comp used reaction
# x[24, i]= if 1<x[9, i]<= 5 then x[9, i]-0.5 -round(x[5, i]/200) else 5 # comp rebuy reaction
# x[11, i]= round(max(0, np.random.uniform(5, 15)-x[7, i] +x[2, i]/4 +x[22, i]/4 )) # own sales new
# x[12, i]= round(min(x[1, i],  max(0, np.random.uniform(5, 15)-x[8, i] +x[3, i]/4 +x[23, i]/4))) # own sales used
# x[13, i]= round(min(x[6, i]/2, max(0, np.random.uniform(5, 15)+x[9, i] -x[4, i]/4 -x[24, i]/4))) # own repurchases
# x[14, i]= x[5, i]*0.05# comp holding cost
# x[15, i]= round(max(0, np.random.uniform(5, 15)-x[2, i] +x[7, i]/4 +x[7, i-1]/4 )) # comp sales new
# x[16, i]= round(min(x[5, i],  max(0, np.random.uniform(5, 15)-x[3, i] +x[8, i]/4 +x[8, i-1]/4 ))) # comp sales used
# x[17, i]= round(min(x[6, i]/2, max(0, np.random.uniform(5, 15)+x[4, i] -x[9, i]/4 -x[9, i-1]/4 ))) # comp repurchases

# x[18, i]= -x[10, i] +x[11, i]*(x[7, i]-c_new) +x[12, i]*x[8, i] -x[13, i]*x[9, i] # own total rewards
# x[19, i]= -x[14, i] +x[15, i]*(x[2, i]-c_new) +x[16, i]*x[3, i] -x[17, i]*x[4, i] # comp total rewards
# x[20, i]= x[18, i]+ if i>0 then x[20, i-1] # own total accumulated rewards
# x[21, i]= x[19, i]+ if i>0 then x[21, i-1] # comp total accumulated rewards

# #######  OLS y1 (own sales new)

y1 = [x[11, i] for i in range(N)]
print('y1:', y1)
y1_df = pd.DataFrame(y1)
y1_df.to_csv('y1_first_prototype.csv')
M1 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)
# param b1 {k in M1} default 0
b1 = np.zeros(27)
# var beta1 {k in M1}
# beta1 = somehow do the beta error
x_y1 = x[M1, :]

A = np.vstack([x_y1, np.ones(len(x_y1))]).T

# minimize OLSy1: sum{i in 1..N} ( sum{k in M1} beta1[k]*x[k,i] - y1[i] )^2
# ols_y1 = np.sum((np.dot(b1[M1], x[M1,:]) - y1)**2) #########################################################################
# y1_t = torch.tensor(y1)

# y_y1_t = torch.tensor(x_y1
# print(y1_t.dim())
# x_y1i = torch.transpose(y1_t, -1, 0)
# y1_t1 = y1_t[:, np.newaxis]


# print("y1_t:", y1_t.size())
print('x_y1:', x_y1.size())
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), y1)
print('alpha:', alpha)
# result = torch.linalg.lstsq(x_y1, y1_t)
# print('result:', result)
# objective OLSy1 solve for{k in M1} let b1[k] = beta1[k]
