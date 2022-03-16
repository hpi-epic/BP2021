import os

import alpha_business.configuration.utils as ut


class PathManager():
	data_path = None

	def manage_data_path(cls, new_path: str) -> None:
		"""
		Manage the data path.
		Used by `main.py` with the `--datapath` argument.

		If the provided new_path is None, check that a valid path is already saved, otherwise overwrite it.

		Args:
			new_path (str | None): The path that should be set.
		"""
		# Make sure the file where we save the path exists
		if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data_path.txt')):
			with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'w') as path_file:
				pass

		# No path provided, check if a valid path is already saved
		if new_path is None:
			with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'r') as path_file:
				old_path = path_file.read()

			assert old_path != '', 'No data path was saved and no data path was provided. Please provide the `--datapath` argument before proceeding'
			# There is a valid path saved
			if ut.readable_dir(old_path):
				print(f'Data will be read from and saved to "{old_path}"')
				return
			# A path has previously been saved, but is no longer a valid directory
			raise AssertionError(f'The current saved data path is invalid: {old_path}\nPlease update it using the `--datapath` argument')

		# A path was provided, but is not valid
		elif not ut.readable_dir(new_path):
			raise AssertionError(f'The provided path is not a valid directory: {new_path}')

		# Valid path provided
		else:
			cls._update_path_file(cls, new_path)

	def _update_path_file(cls, new_path: str) -> None:
		"""
		Update the path file and instance variable with the new path.

		Args:
			new_path (str): The data path to be saved.
		"""
		cls.data_path = new_path

		with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'w') as path_file:
			path_file.write(cls.data_path)

		print(f'Data will be read from and saved to "{cls.data_path}"')


if __name__ == '__main__':
	PathManager._get_data_path(PathManager)
	PathManager.change_data_path(PathManager)
