import numpy as np

import recommerce.configuration.environment_config as config
import recommerce.configuration.utils as ut


class Watcher:
	"""
	This class administrates all collectable information about the success and behaviour of the agent.
	"""
	def __init__(self, number_envs: int = 1):
		self.number_envs = number_envs
		self.all_dicts = []
		if number_envs == 1:
			self.step_counter = 0
			self.cumulated_info = None
		else:
			self.step_counters = [0 for _ in range(number_envs)]
			self.info_accumulators = [None for _ in range(number_envs)]

	def add_info(self, info: dict, index: int = None):
		if self.number_envs == 1:
			assert index is None
			self.cumulated_info = info if self.cumulated_info is None else ut.add_content_of_two_dicts(self.cumulated_info, info)
			self.step_counter += 1
			assert self.step_counter <= config.episode_length
		else:
			assert index is not None
			assert index < self.number_envs and index >= 0
			self.info_accumulators[index] = \
				info if self.info_accumulators[index] is None else ut.add_content_of_two_dicts(self.info_accumulators[index], info)
			self.step_counters[index] += 1
			assert self.step_counters[index] <= config.episode_length

	def finish_episode(self, index: int = None):
		if self.number_envs == 1:
			assert index is None
			assert self.step_counter == config.episode_length
			self.all_dicts.append(self.cumulated_info)
			self.cumulated_info = None
			self.step_counter = 0
		else:
			assert index is not None
			assert index < self.number_envs and index >= 0
			assert self.step_counters[index] == config.episode_length
			self.all_dicts.append(self.info_accumulators[index])
			self.info_accumulators[index] = None
			self.step_counters[index] = 0

	def get_average_dict(self, look_back: int = 100) -> dict:
		slice_dicts = self.all_dicts[-look_back:]
		averaged_info = slice_dicts[0]
		for i, next_dict in enumerate(slice_dicts):
			if i != 0:
				averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
		averaged_info = ut.divide_content_of_dict(averaged_info, len(slice_dicts))
		return averaged_info

	def get_all_samples_of_property(self, property_name: str, vendor: int) -> list:
		return [mydict[property_name][f'vendor_{vendor}'] for mydict in self.all_dicts]

	def get_progress_values_of_property(self, property_name: str, vendor: int, look_back: int = 100) -> list:
		progress_values = [mydict[property_name][f'vendor_{vendor}'] for mydict in self.all_dicts]
		return [np.mean(progress_values[max(i-look_back, 0):i]) for i in range(len(progress_values))]
