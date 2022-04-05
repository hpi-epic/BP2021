# #########  Simulation in Modell B    ###############
import numpy as np

N = 200

x = np.array([[(1 if k == 0 else np.round_(np.random.uniform(0, 20))) for k in range(N)] for _ in range(25)])  # historical/training data

NB = N
M1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
M2 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
M3 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
M4 = (1, 2, 3)
M4x = (1, 2, 4)
M5 = (1, 2, 3, 4)
M5x = (1, 2, 4, 5)
M6 = (1, 2, 3, 4, 5)
M6x = (1, 2, 4, 5, 6)
b1 = (1, 2, 3, 4, 5, 6)
b2 = (1, 2)
b3 = (1, 2, 3)
b4 = [1, 2, 3, 4]
b4x = [1, 2, 4, 8]
b5 = [1, 2, 3, 4, 5]
b5x = [1, 2, 4, 8, 16]
b6 = [1, 2, 3, 4, 5, 6]
b6x = [1, 2, 4, 8, 16, 32]
c_new = None
# param xb{k in 0..24,i in 0..NB} default if k=0 then 1 else if i=0 then 5 else -1      # simulated data

xb = np.array([[(1 if k == 0 else 5 if i == 0 else -1) for k in range(0, 24)] for i in range(0, NB)])

for i in range(1, NB):
	xb[1, i] = xb[1, i-1]-xb[12, i-1]+xb[13, i-1]

	# xb[2,i] = sum{k in M4}  b4[k] *xb[k,i-1] + sum{k in M4x} (b4x[k]*(1 if xb[7,i-1]<k else 0)) for k in M4x
	# xb[2,i] = [(b4[k] *xb[k,i-1] + (b4x[k2]*(1 if xb[7,i-1]<k2 else 0)) for k2 in M4x) for k in M4]  # comp price new
	for k in M4:
		xb[2, i] += b4[k] * xb[k, i-1]
	for k in M4x:
		xb[2, i] += (b4x[k]*(1 if xb[7, i-1] < k else 0))  # comp price new

	for k in M5:
		xb[3, i] += b5[k] * xb[k, i-1]
	for k in M5x:
		xb[3, i] += b5x[k]*(1 if xb[8, i-1] < k else 0)  # comp price used
	xb[3, i]
	# xb[3,i] = sum{k in M5}  b5[k] *xb[k,i-1] + sum{k in M5x} b5x[k]*(if xb[8,i-1]<k then 1 else 0)  # comp price used

	xb[4, i] = sum([b6[k] * xb[k, i-1] for k in M6]) + sum([b6x[k]*(1 if xb[9, i-1] < k else 0) for k in M6x])
	# xb[4,i] = sum{k in M6}  b6[k] *xb[k,i-1] + sum{k in M6x} b6x[k]*(if xb[9,i-1]<k then 1 else 0)  # comp price rebuy

	xb[5, i] = xb[5, i-1]-xb[16, i-1]+xb[17, i-1]  # comp inventory

	xb[6, i] = max(0, np.round_(0.8*xb[6, i-1])+xb[11, i-1]+xb[12, i-1]-xb[13, i-1]+xb[15, i-1]+xb[16, i-1]-xb[17, i-1])  # resources in use

	xb[7, i] = xb[2, i] - 1 + np.round_(x[1, i]/200) if 9 < xb[2, i] <= 18 else 18  # own price new
	xb[8, i] = xb[3, i] - 1 + np.round_(x[1, i]/200) if 5 < xb[3, i] <= 12 else 12  # own price used
	xb[9, i] = xb[4, i] - 0.5 - np.round_(x[1, i]/200) if 1 < xb[4, i] <= 5 else 5  # own price rebuy
	xb[10, i] = xb[1, i]*0.05  # own holding cost

	# xb[22,i]= sum{k in M4}  b4[k] *xb[k,i] + sum{k in M4x} b4x[k]*(if xb[7,i]<k then 1 else 0)  # comp new reaction
	for k in M4:
		xb[22, i] += b4[k] * xb[k, i]
	for k in M4x:
		b4x[k] * (1 if xb[7, i] < k else 0)

	# xb[23,i]= sum{k in M5}  b5[k] *xb[k,i] + sum{k in M5x} b5x[k]*(if xb[8,i]<k then 1 else 0)  # comp used reaction

	for k in M5:
		xb[23, i] += b6[k] * xb[k, i]
	for k in M5x:
		xb[23, i] += b6x[k]*(1 if xb[8, i] < k else 0)

	# xb[24,i]= sum{k in M6}  b6[k] *xb[k,i] + sum{k in M6x} b6x[k]*(if xb[9,i]<k then 1 else 0)  # comp rebuy reaction
	for k in M6:
		xb[24, i] += b6[k] * xb[k, i]
	for k in M6x:
		xb[24, i] += b6x[k]*(1 if xb[9, i] < k else 0)

	# xb[11,i]= np.round_(max(0, np.random.uniform(-5,5) + sum{k in M1} b1[k]*xb[k,i]))
	xb[11, i] = np.round_(max(np.random.uniform(-5, 5) + sum([b1[k] * xb[k, i] for k in M1])))

	# xb[12,i]= np.round_(min(x[1,i],  max(0,np.random.uniform(-5,5) + sum{k in M2} b2[k]*xb[k,i])))
	xb[12, i] = np.round_(min(x[1, i], max(np.random.uniform(-5, 5) + sum([b2[k] * xb[k, i] for k in M2]), 0)))
	# xb[13,i]= np.round_(min(x[6,i]/2,max(0,np.random.uniform(-5,5) + sum{k in M3} b3[k]*xb[k,i])))
	xb[13, i] = np.round_(min(x[6, i] / 2, max(np.random.uniform(-5, 5) + sum([b3[k] * xb[k, i] for k in M3]), 0)))
	xb[14, i] = xb[5, i]*0.05

	xb[15, i] = np.round_(max(0, np.random.uniform(-5, 5) + b1[0] * xb[0, i] + b1[1] * xb[5, i] + b1[2] * xb[7, i-1] + b1[3] * xb[8, i-1]
		+ b1[4] * xb[9, i-1] + b1[7] * xb[2, i] + b1[8] * xb[3, i] + b1[9] * xb[4, i] + b1[22] * xb[7, i] + b1[23] * xb[8, i]
		+ b1[24] * xb[9, i]))  # cf M1

	xb[16, i] = np.round_(min(x[5, i],  max(0, np.random.uniform(-5, 5) + b2[0] * xb[0, i] + b2[1] * xb[5, i] + b2[2] * xb[7, i-1]
		+ b2[3] * xb[8, i - 1]
		+ b2[4] * xb[9, i-1] + b2[7] * xb[2, i] + b2[8] * xb[3, i] + b2[9] * xb[4, i] + b2[22] * xb[7, i] + b2[23] * xb[8, i]
		+ b2[24] * xb[9, i])))  # cf M2

	xb[17, i] = np.round_(min(x[6, i]/2, max(0, np.np.random.uniform(-5, 5) + b3[0] * xb[0, i] + b3[1] * xb[5, i] + b3[2] * xb[7, i-1]
		+ b3[3] * xb[8, i-1] + b3[4] * xb[9, i-1] + b3[7] * xb[2, i] + b3[8] * xb[3, i] + b3[9] * xb[4, i] + b3[22] * xb[7, i]
		+ b3[23] * xb[8, i] + b3[24] * xb[9, i])))  # cf M3

	xb[18, i] = -xb[10, i]+xb[11, i]*(xb[7, i] - c_new)+xb[12, i]*xb[8, i]-xb[13, i]*xb[9, i]      # own total rewards
	xb[19, i] = -xb[14, i]+xb[15, i]*(xb[2, i] - c_new)+xb[16, i]*xb[3, i]-xb[17, i]*xb[4, i]      # comp total rewards
	xb[20, i] = xb[18, i] + (xb[20, i-1] if i > 0 else 0)  # own total accumulated rewards
	xb[21, i] = xb[19, i] + (xb[21, i-1] if i > 0 else 0)  # comp total accumulated rewards
