import numpy as np
from attrdict import AttrDict

import recommerce.configuration.utils as ut


class Watcher:
	"""
	This class administrates all collectable information about the success and behaviour of the agent.
	"""
	def __init__(self, config: AttrDict, number_envs: int = 1):
		self.number_envs = number_envs
		self.all_dicts = []
		self.step_counters = [0 for _ in range(number_envs)]
		self.info_accumulators = [None for _ in range(number_envs)]
		self.config = config

	def add_info(self, info: dict, index: int = 0):
		"""
		Add a info dict received from the environment to the accumulator.

		Args:
			info (dict): A raw dict containing the information from the environment.
			index (int, optional): If there are multiple environments in parallel, this index is used to specify which environment was used.
			Defaults to 0.
		"""
		assert index < self.number_envs and index >= 0
		self.info_accumulators[index] = \
			info if self.info_accumulators[index] is None else ut.add_content_of_two_dicts(self.info_accumulators[index], info)
		self.step_counters[index] += 1
		assert self.step_counters[index] <= self.config.episode_length
		if self.step_counters[index] == self.config.episode_length:
			self._finish_episode(index)

	def _finish_episode(self, index: int = 0):
		"""
		As soon as one episode is finished, the entries of this episode are added to the list of all dicts.
		"""
		assert index < self.number_envs and index >= 0
		assert self.step_counters[index] == self.config.episode_length
		self.all_dicts.append(self.info_accumulators[index])
		self.info_accumulators[index] = None
		self.step_counters[index] = 0

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

	def get_all_samples_of_property(self, property_name: str, vendor: int or None) -> list:
		"""
		Returns all values of a property in the sequence they have been saved.

		Args:
			property_name (str): The requested property.
			vendor (int or None): The vendor if this property is saved for multiple vendors. It is None if it is a global property.
			Defaults to None.

		Returns:
			list: The list containing all values of the property.
		"""
		if vendor is None:
			return [tmp_dict[property_name] for tmp_dict in self.all_dicts]
		else:
			return [tmp_dict[property_name][f'vendor_{vendor}'] for tmp_dict in self.all_dicts]

	def get_cumulative_properties(self) -> dict:
		"""
		The watcher saves all steps of the episode with in dicts, one for each step.
		But if you want to get samples to calculate with or to plot, you need all values of one property in one list.
		This method returns all values which can be cumulated sensibly in a dict with one entry per property.
		These entries will contain a list of all values of the property in the sequence they have been saved.

		Returns:
			dict: A dict containing a list of all values for each property.
		"""
		output_dict = {}
		for key in sorted(list(self.all_dicts[0].keys())):
			if key.startswith('state') or key.startswith('actions'):
				continue  # skip these properties because they are not cumulative

			if isinstance(self.all_dicts[0][key], dict):
				output_dict[key] = [self.get_all_samples_of_property(key, vendor) for vendor in range(self.get_number_of_vendors())]
			else:
				output_dict[key] = self.get_all_samples_of_property(key, None)

		return output_dict

	def get_progress_values_of_property(self, property_name: str, vendor: int or None = None, look_back: int = 100) -> list:
		"""
		During training, mean values shall be displayed to assess the progress of the agent.
		Because the agent's policy changes and nearly smooth values should be displayed, a rolling average is a compromise.
		Other solutions would need extremely more computational power.

		Args:
			property_name (str): The name of the property to get the rolling average of.
			vendor (int or None): The vendor you want to access. Should be None if this is a global property. Defaults to None.
			look_back (int, optional): The number of steps the rolling average should calculate back. Defaults to 100.

		Returns:
			list: The list containing the rolling averages over the past episode.
		"""
		if vendor is None:
			progress_values = [tmp_dict[property_name] for tmp_dict in self.all_dicts]
		else:
			progress_values = [tmp_dict[property_name][f'vendor_{vendor}'] for tmp_dict in self.all_dicts]
		return [np.mean(progress_values[max(i-look_back, 0):(i + 1)]) for i in range(len(progress_values))]

	def get_number_of_vendors(self) -> int:
		return len(self.all_dicts[0]['profits/all'])
