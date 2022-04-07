from typing import Tuple

import gym
import numpy as np
import torch

NB = 200


class SimMarketKalibrated(gym.Env):
	cost_new_product = 3

	def __init__(self, by1, by2, by3, by4, bxy4, by5, bxy5, by6, bxy6, M1, M2, M3, M4, M5, M6, M4x, M5x, M6x) -> None:
		self.by1 = by1
		self.by2 = by2
		self.by3 = by3
		self.by4 = by4
		self.bxy4 = bxy4
		self.by5 = by5
		self.bxy5 = bxy5
		self.by6 = by6
		self.bxy6 = bxy6
		self.M1 = M1
		self.M2 = M2
		self.M3 = M3
		self.M4 = M4
		self.M5 = M5
		self.M6 = M6
		self.M4x = M4x
		self.M5x = M5x
		self.M6x = M6x

	def comp_prices(Mi, Mix, bi, bix, flag: str, xb, state_to_get_parameters_from):
		xb_index = -1
		if flag == 'new':
			xb_index = 7
		elif flag == 'used':
			xb_index = 8
		elif flag == 'rebuy':
			xb_index = 9
		else:
			assert False

		# xb[4,i] = sum{k in M6}  b6[k] *xb[k,i-1] + sum{k in M6x} b6x[k]*(if xb[9,i-1]<k then 1 else 0)  # comp price rebuy (old)
		# xb[24,i]= sum{k in M6}  b6[k] *xb[k,i] + sum{k in M6x} b6x[k]*(if xb[9,i]<k then 1 else 0)  # comp price rebuy (updated)
		return sum([bi[ki] * state_to_get_parameters_from[k] for ki, k in enumerate(Mi)]) + \
			sum([bix[ki] * (1 if state_to_get_parameters_from[xb_index] < k else 0) for ki, k in enumerate(Mix)])

	def reset() -> np.array:
		observable_state = (6, 1, 22, 23, 24)
		market_state = np.array([-1.0000e+00,  2.3400e+02,  2.2745e+01,  1.3406e+01,  2.5241e+00,
		2.7200e+02,  2.7000e+01,  1.8000e+01,  1.2000e+01,  2.4086e-02,
		1.1700e+01,  2.0000e+00,  0.0000e+00,  0.0000e+00,  1.3600e+01,
		3.0000e+00,  0.0000e+00,  3.0000e+00,  1.6300e+01,  3.5062e+01,
		1.0208e+04,  2.3988e+03,  1.8651e+01,  1.2789e+01,  4.5772e+00])
		return np.array([market_state[state_index] for state_index in observable_state])

	def step(self, agent_action, previous_state) -> Tuple[np.array, np.float32, bool, dict]:
		xb = torch.tensor([[(1. if k-1 == 0 else 5. if i == 0 else -1.) for k in range(0, 25)] for i in range(0, NB)]).transpose(0, 1)
		xb[1] = previous_state[1] - previous_state[12] + previous_state[13]  # agent inventory (after the previous step)

		xb[2] = self.comp_prices(self.Mb, self.M4x, self.b4, self.b4x, 'new', xb, previous_state)  # comp price new 		(old)
		xb[3] = self.comp_prices(self.Mb, self.M5x, self.b5, self.b5x, 'used', xb, previous_state)  # comp price used 	(old)
		xb[4] = self.comp_prices(self.Mb, self.M6x, self.b6, self.b6x, 'rebuy', xb, previous_state)  # comp price rebuy 	(old)

		xb[5] = previous_state[5] - previous_state[16] + previous_state[17]  # comp inventory (after the previous step)

		xb[6] = max(0, np.round_(0.8*previous_state[6]) + previous_state[11] + previous_state[12] - previous_state[13]
			+ previous_state[15] + previous_state[16] - previous_state[17])  # resources in use (after the previous step)

		# TODO: check if there is a logic behind this
		# let xb[7,i] := if 9<xb[2,i]<=18 then xb[2,i]-1  +round(x[1,i]/200) else 18;
		# let xb[8,i] := if 5<xb[3,i]<=12 then xb[3,i]-1  +round(x[1,i]/200) else 12;
		# let xb[9,i] := if 1<xb[4,i]<= 5 then xb[4,i]-0.5-round(x[1,i]/200) else  5;
		# xb[7] = xb[2] - 1 + np.round_(x[1]/200) if 9 < xb[2] <= 18 else 18  # agent price new
		# xb[8] = xb[3] - 1 + np.round_(x[1]/200) if 5 < xb[3] <= 12 else 12  # agent price used
		# xb[9] = xb[4] - 0.5 - np.round_(x[1]/200) if 1 < xb[4] <= 5 else 5    # agent price rebuy

		xb[7] = agent_action[0]
		xb[8] = agent_action[1]
		xb[9] = agent_action[2]

		xb[10] = xb[1] * 0.05  # agent holding cost

		xb[22] = self.comp_prices(self.Mb, self.M4x, self.b4, self.b4x, 'new', xb, xb)  # comp price new 		(updated)
		xb[23] = self.comp_prices(self.Mb, self.M5x, self.b5, self.b5x, 'used', xb, xb)  # comp price used 	(updated)
		xb[24] = self.comp_prices(self.Mb, self.M6x, self.b6, self.b6x, 'rebuy', xb, xb)  # comp price rebuy 	(updated)

		# xb[11,i]= np.round_(max(0, np.random.uniform(-5,5) + sum{k in self.Ma} self.b1[k]*xb[k,i]))

		xb[11] = np.round_(max(np.random.uniform(-5, 5) + sum([self.b1[ki] * xb[k] for ki, k in enumerate(self.Ma)]), 0))
		# agent sales new
		xb[12] = np.round_(min(xb[1], max(np.random.uniform(-5, 5) + sum([self.b2[ki] * xb[k] for ki, k in enumerate(self.Ma)]), 0)))
		# agent sales used
		xb[13] = np.round_(min(xb[6] / 2, max(np.random.uniform(-5, 5) + sum([self.b3[ki] * xb[k] for ki, k in enumerate(self.Ma)]), 0)))
		# agent sales rebuy

		xb[14] = xb[5]*0.05
		# TODO: refactor these into loops or something more beautiful
		xb[15] = np.round_(max(0, np.random.uniform(-5, 5)
			+ self.b1[0] * xb[0] + self.b1[1] * xb[5] + self.b1[2] * previous_state[7]
			+ self.b1[3] * previous_state[8] + self.b1[4] * previous_state[9] + self.b1[5] * xb[2] + self.b1[6] * xb[3]
			+ self.b1[7] * xb[4] + self.b1[8] * xb[7] + self.b1[9] * xb[8] + self.b1[10] * xb[9]))  # cf self.Ma # competitor sales new

		xb[16] = np.round_(min(xb[5],  max(0, np.random.uniform(-5, 5)
			+ self.b2[0] * xb[0] + self.b2[1] * xb[5] + self.b2[2] * previous_state[7]
			+ self.b2[3] * previous_state[8] + self.b2[4] * previous_state[9] + self.b2[5] * xb[2] + self.b2[6] * xb[3]
			+ self.b2[7] * xb[4] + self.b2[8] * xb[7] + self.b2[9] * xb[8] + self.b2[10] * xb[9])))  # cf self.Ma # competitor sales used

		xb[17] = np.round_(min(xb[6] / 2, max(0, np.random.uniform(-5, 5)
			+ self.b3[0] * xb[0] + self.b3[1] * xb[5] + self.b3[2] * previous_state[7]
			+ self.b3[3] * previous_state[8] + self.b3[4] * previous_state[9] + self.b3[5] * xb[2] + self.b3[6] * xb[3]
			+ self.b3[7] * xb[4] + self.b3[8] * xb[7]
			+ self.b3[9] * xb[8] + self.b3[10] * xb[9])))  # cf self.Ma # competitor sales rebuy

		# rewards
		xb[18] = -xb[10] + xb[11] * (xb[7] - self.cost_new_product) + xb[12] * xb[8] - xb[13] * xb[9]  # agent  total rewards
		xb[19] = -xb[14] + xb[15] * (xb[2] - self.cost_new_product) + xb[16] * xb[3] - xb[17] * xb[4]  # comp total rewards

		# rewards cumulated
		xb[20] = xb[18] + (previous_state[20] if previous_state is not None else 0)  # agent total accumulated rewards
		xb[21] = xb[19] + (previous_state[21] if previous_state is not None else 0)  # comp total accumulated rewards
		observable_state = (6, 1, 22, 23, 24)
		agent_observation = np.array([xb[state_index] for state_index in observable_state])
		return agent_observation, xb[18], False, {}
