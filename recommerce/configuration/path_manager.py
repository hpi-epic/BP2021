import os


# DO NOT move this to utils.py, this will create a circular import and break the application!
def is_readable_dir(path) -> bool:
	"""
	Helper function defining whether or not a path is an existing and readable path.

	Adapted from https://stackoverflow.com/questions/11415570/directory-path-types-with-argparse

	Args:
		path (str): The path to check.

	Returns:
		bool: If the path exists and is readable.
	"""
	return bool(os.access(path, os.R_OK)) if os.path.isdir(path) else False


def get_user_path() -> str:
	"""
	Helper function that reads the user_path.txt and return its path.

	Returns:
		str: The data path.
	"""
	with open(os.path.join(os.path.dirname(__file__), 'user_path.txt'), 'r') as path_file:
		return path_file.read()


class PathManager():
	# Since we want to allow devs to call different parts of the application without going through `main.py` we need a way of getting
	# the correct data path without havign to call `manage_data_path`, which is why we use `get_user_path()`.
	# Note: The `PathManager` is smart and knows if it has already called `get_user_path()`, so each time the program is run it will
	# only get called once, even when `PathManager.user_path` is used multiple times.
	user_path = get_user_path()
	results_path = os.path.join(user_path, 'results')
	data_path = os.path.join(user_path, 'data')

	def manage_user_path(cls, new_path: str) -> None:
		"""
		Manage the data path.
		Used by `main.py` with the `--datapath` argument.

		If the provided new_path is None, check that a valid path is already saved, otherwise overwrite it.

		Args:
			new_path (str | None): The path that should be set.
		"""
		# Make sure the file where we save the path exists
		if not os.path.exists(os.path.join(os.path.dirname(__file__), 'user_path.txt')):
			with open(os.path.join(os.path.dirname(__file__), 'user_path.txt'), 'w') as path_file:
				pass

		# No path provided, check if a valid path is already saved
		if new_path is None:
			with open(os.path.join(os.path.dirname(__file__), 'user_path.txt'), 'r') as path_file:
				old_path = path_file.read()

			assert old_path != '', 'Please provide the `--datapath` argument before proceeding. Use "." to use the current directory'
			# There is a valid path saved
			if is_readable_dir(old_path):
				print(f'Data will be read from and saved to "{old_path}"')
				return
			# A path has previously been saved, but is no longer a valid directory
			raise AssertionError(f'The current saved data path is invalid: {old_path}\nPlease update it using the `--datapath` argument')

		# A path was provided, but is not valid
		elif not is_readable_dir(new_path):
			raise AssertionError(f'The provided path is not a valid directory: {new_path}')

		# Valid path provided
		else:
			cls._update_path_file(cls, new_path)

	def _update_path_file(cls, new_path: str) -> None:
		"""
		Update the path file with the new path.

		Args:
			new_path (str): The data path to be saved.
		"""
		with open(os.path.join(os.path.dirname(__file__), 'user_path.txt'), 'w') as path_file:
			path_file.write(os.path.abspath(new_path))

		cls.user_path = os.path.abspath(new_path)

		print(f'Data will be read from and saved to "{cls.user_path}"')


if __name__ == '__main__':
	print('Looking to update your datapath? Use `recommerce --datapath "YourDatapathHere"`!')
