import numpy as np

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import config


class Watcher:
	"""
	This class administrates all collectable information about the success and behaviour of the agent.
	"""
	def __init__(self, number_envs: int = 1):
		self.number_envs = number_envs
		self.all_dicts = []
		if self.number_envs == 1:
			self.step_counter = 0
			self.cumulated_info = None
		else:
			self.step_counter = [0 for _ in range(number_envs)]
			self.info_accumulators = [None for _ in range(number_envs)]

	def add_info(self, info: dict, index: int = None):
		if self.number_envs == 1:
			assert index is None
			self.cumulated_info = info if self.cumulated_info is None else ut.add_content_of_two_dicts(self.cumulated_info, info)
			self.step_counter += 1
			assert self.step_counter <= config.episode_length
			if self.step_counter == config.episode_length:
				self.finish_episode()
		else:
			assert index is not None
			assert index < self.number_envs and index >= 0
			self.info_accumulators[index] = \
				info if self.info_accumulators[index] is None else ut.add_content_of_two_dicts(self.info_accumulators[index], info)
			self.step_counter[index] += 1
			assert self.step_counter[index] <= config.episode_length
			if self.step_counter[index] == config.episode_length:
				self.finish_episode(index)

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
			assert self.step_counter[index] == config.episode_length
			self.all_dicts.append(self.info_accumulators[index])
			self.info_accumulators[index] = None
			self.step_counter[index] = 0

	def get_average_dict(self, look_back: int = 100) -> dict:
		"""
		Takes a list of dictionaries and calculates the average over the last look_back episodes.
		Assumes that all dicts have the same shape.

		Args:
			look_back (int): The number of episode dictionaries to average over.

		Returns:
			dict: A dict of the same shape containing the average in each entry.
		"""
		if len(self.all_dicts) == 0:
			return {}
		slice_dicts = self.all_dicts[-look_back:]
		averaged_info = slice_dicts[0]
		for i, next_dict in enumerate(slice_dicts):
			if i != 0:
				averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
		averaged_info = ut.divide_content_of_dict(averaged_info, len(slice_dicts))
		return averaged_info

	def get_all_samples_of_property(self, property_name: str, vendor: int) -> list:
		if vendor is None:
			return [mydict[property_name] for mydict in self.all_dicts]
		else:
			return [mydict[property_name][f'vendor_{vendor}'] for mydict in self.all_dicts]

	def get_cumulative_properties(self) -> dict:
		output_dict = {}
		for key in sorted(list(self.all_dicts[0].keys())):
			if key.startswith('state') or key.startswith('actions'):
				continue  # skip these properties because they are not cumulative

			if isinstance(self.all_dicts[0][key], dict):
				for vendor in range(self.get_number_of_vendors()):
					output_dict[f'{key}/vendor_{vendor}'] = self.get_all_samples_of_property(key, vendor)
			else:
				output_dict[key] = self.get_all_samples_of_property(key, None)

		return output_dict

	def get_progress_values_of_property(self, property_name: str, vendor: int, look_back: int = 100) -> list:
		progress_values = [mydict[property_name][f'vendor_{vendor}'] for mydict in self.all_dicts]
		return [np.mean(progress_values[max(i-look_back, 0):i]) for i in range(len(progress_values))]

	def get_number_of_vendors(self) -> int:
		return len(self.all_dicts[0]['profits/all'])
