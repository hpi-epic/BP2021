# import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from recommerce.configuration.path_manager import PathManager
from recommerce.market.sim_market_kalibrated import SimMarketKalibrated

# import pandas as pd
# import torch


class KalibratedDatagenerator(SimMarketKalibrated):

	def __init__(self, config, support_continuous_action_space: bool = False) -> None:
		super(KalibratedDatagenerator, self).__init__(config, support_continuous_action_space)
		self.cumulated_states = np.array(np.zeros(25)).reshape(25, 1)
		self.episode_counter = 0
		self.is_tracking = False
		# print(self.cumulated_states.shape)
		# self.cumulated_states = self.cumulated_states

	# def step(self, action) -> Tuple[np.array, np.float32, bool, dict]:
	# 	step_output_tuple = super(CircularEconomyDatagenerator, self).step(action)
	# 	self.is_tracking = False

	# 	if self.is_tracking and self.episode_counter < 6000:
	# 		self.output_dict_append(self._output_dict, self.step_counter)
	# 	elif self.episode_counter >= 5000 and not self.is_tracking:
	# 		self.is_tracking = True
	# 		# print('*** start tracking ***')
	# 		# print()
	# 		self.output_dict_append(self._output_dict, self.step_counter)

	# 	self.episode_counter += 1

	def step(self, action) -> Tuple[np.array, np.float32, bool, dict]:
		step_output_tuple = super(KalibratedDatagenerator, self).step(action)
		self.is_tracking = True

		# if self.is_tracking and self.episode_counter < 6000:
		self.output_dict_append(self._output_dict, self.step_counter)
		# elif self.episode_counter >= 5000 and not self.is_tracking:
		# 	self.is_tracking = True
		# 	print('*** start tracking ***')
		# 	print()
		# 	self.output_dict_append(self._output_dict, self.step_counter)

		return step_output_tuple

	def output_dict_append(self, output_dict, i):
		x = self.xb.reshape(25, 1)  # not a dictionary

		self.cumulated_states = np.hstack((self.cumulated_states, np.round(x, 3)))

		self.episode_counter += 1
		# self.rainer_states = np.hstack((self.rainer_states, np.round(x, 3)))

		# print('episode_counter: ', self.episode_counter)
		if self.episode_counter == self.config.episode_length * 10:
			saving_array = self.cumulated_states.transpose()
			# saving_array_rainer = self.rainer_states.transpose()
			print(saving_array)
			# Saving the array in a text file

			# np.savetxt(f'{PathManager.data_path}/kalibration_data/training_data-txtfile_int1dot.txt',
			# 	saving_array_rainer, delimiter=' ', fmt='%1.3f', newline='\n')

			save_path = f'{PathManager.data_path}/comparison_data/comparison_data_dnn1.csv'
			df = pd.DataFrame(saving_array)
			print(df.head())
			df.to_csv(save_path, index=False)
			print(f'data saved to {save_path}')
			exit(0)

		# self.cumulated_states = np.hstack((self.cumulated_states, x))
		# print('episode_counter: ', self.episode_counter)
		# if self.episode_counter == 5999:

		# 	pd.DataFrame(self.cumulated_states).to_csv(f'{PathManager.data_path}/kalibration_data/training_data_1000.csv', index=False)
		# 	print('data saved')
		# 	exit(0)
