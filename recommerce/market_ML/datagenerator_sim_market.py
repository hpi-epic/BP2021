# import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly

# import pandas as pd
# import torch


class CircularEconomyDatagenerator(CircularEconomyRebuyPriceDuopoly):

	def __init__(self, support_continuous_action_space: bool = False) -> None:
		super(CircularEconomyDatagenerator, self).__init__(support_continuous_action_space)
		self.cumulated_states = np.array(np.zeros(25)).reshape(25, 1)
		self.episode_counter = 0
		self.is_tracking = False
		# print(self.cumulated_states.shape)
		# self.cumulated_states = self.cumulated_states

	def step(self, action) -> Tuple[np.array, np.float32, bool, dict]:
		step_output_tuple = super(CircularEconomyDatagenerator, self).step(action)
		self.is_tracking = False

		if self.is_tracking and self.episode_counter < 6000:
			self.output_dict_append(self._output_dict, self.step_counter)
		elif self.episode_counter >= 5000 and not self.is_tracking:
			self.is_tracking = True
			print('*** start tracking ***')
			print()
			self.output_dict_append(self._output_dict, self.step_counter)

		self.episode_counter += 1

		return step_output_tuple

	def output_dict_append(self, output_dict: dict, i):
		x = np.array(np.zeros(25)).reshape(25, 1)
		print('i: ', i)
		# print('cumulated_states: ', self.cumulated_states)
		# agent period: 1st half: agent x[7]vs x[2]| 2nd half: agent x[7] vs x[2, i+1]
		# comp period: 1st half: com x[2, i-1] vs x[7, i-1] | 2nd half: com x[2, i-1] vs x[7]

		x[0, 0] = i
		assert float(output_dict['state/in_storage']['vendor_0']).is_integer()
		x[1, 0] = int(output_dict['state/in_storage']['vendor_0'])  # agent inventory
		if i == 1:
			x[2, 0] = self.cumulated_states[22, 0]  # comp price new (old)
			x[3, 0] = self.cumulated_states[23, 0]   # comp price used (old)
			x[4, 0] = self.cumulated_states[24, 0]   # comp price rebuy (old)
		else:

			x[2, 0] = self.cumulated_states[22, i-1]  # comp price new (old)
			x[3, 0] = self.cumulated_states[23, i-1]   # comp price used (old)
			x[4, 0] = self.cumulated_states[24, i-1]   # comp price rebuy (old)

		x[5, 0] = output_dict['state/in_storage']['vendor_1']  # comp inventory
		assert float(output_dict['state/in_circulation']).is_integer()
		x[6, 0] = int(output_dict['state/in_circulation'])  # resource in use

		x[7, 0] = output_dict['actions/price_refurbished']['vendor_0']  # agent price new
		x[8, 0] = output_dict['actions/price_new']['vendor_0']  # agent price used
		x[9, 0] = output_dict['actions/price_new']['vendor_0']  # agent price rebuy

		x[10, 0] = output_dict['profits/storage_cost']['vendor_0']  # agent storage cost

		assert float(output_dict['actions/price_new']['vendor_1']).is_integer()
		assert float(output_dict['actions/price_refurbished']['vendor_1']).is_integer()
		assert float(output_dict['actions/price_rebuy']['vendor_1']).is_integer()

		x[22, 0] = int(output_dict['actions/price_new']['vendor_1'])  # comp price new (updated)
		x[23, 0] = int(output_dict['actions/price_refurbished']['vendor_1'])  # comp price used (updated)
		x[24, 0] = int(output_dict['actions/price_rebuy']['vendor_1'])  # comp price rebuy (updated)

		x[11, 0] = output_dict['customer/purchases_new']['vendor_0']  # agent sales new
		x[12, 0] = output_dict['customer/purchases_refurbished']['vendor_0']  # agent sales used
		x[13, 0] = output_dict['owner/rebuys']['vendor_0']  # agent sales rebuy

		x[14, 0] = output_dict['profits/storage_cost']['vendor_1']  # comp holding cost

		x[15, 0] = output_dict['customer/purchases_new']['vendor_1']  # comp sales new
		x[16, 0] = output_dict['customer/purchases_refurbished']['vendor_1']  # comp sales used
		x[17, 0] = output_dict['owner/rebuys']['vendor_1']  # comp sales rebuy

		x[18, 0] = output_dict['profits/all']['vendor_0']  # agent total reward
		x[19, 0] = output_dict['profits/all']['vendor_1']  # comp total reward

		x[20, 0] = self.cumulated_states[20, i-1] + x[18, 0]  # agent culmulated reward
		x[21, 0] = self.cumulated_states[21, i-1] + x[19, 0]  # comp culmulated reward
		# pd.DataFrame(x).to_csv(f'kalibration_data/training_data-{datetime.datetime.now()}.csv', index=False)

		self.cumulated_states = np.hstack((self.cumulated_states, x))
		print('episode_counter: ', self.episode_counter)
		if self.episode_counter == 1100:

			pd.DataFrame(self.cumulated_states).to_csv(f'{PathManager.data_path}/kalibration_data/training_data_1000.csv', index=False)
			print('data saved')
			exit(0)
