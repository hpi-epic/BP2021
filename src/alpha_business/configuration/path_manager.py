import os


class PathManager():
	data_path = None

	def _get_data_path(cls) -> None:
		"""
		Try and load the file where the user has previously saved the path they want to use for data.

		If the user has not yet provided a path, ask them for one and validate it.
		"""
		if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data_path.txt')):
			with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'w') as path_file:
				pass

		with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'r') as path_file:
			data_path = path_file.read()

		# There is a valid path saved
		if os.path.isdir(data_path):
			print(f'Data will be read from and saved to "{data_path}"')
			return
		# A path has previously been saved, but is no longer a valid directory
		elif data_path != '':
			print(f'The current saved data path does not exist: {data_path}')

		# No path provided as of yet
		while not os.path.isdir(data_path):
			data_path = input('Please provide a path where alpha_business will look for and save data:\n')
			if not os.path.isdir(data_path):
				print(f'The provided path is not a valid directory: {data_path}')

		cls.update_data_path(data_path)

	def update_data_path(cls, path: str) -> None:
		"""
		Use this to manually update the data path.
		Used by `main.py` if the `--datapath` argument is set.

		Args:
			path (str): The path that should be set.
		"""
		assert os.path.isdir(path), f'The provided path is not valid: {path}'
		cls.data_path = path

		with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'w') as path_file:
			path_file.write(cls.data_path)

		print(f'Data will be read from and saved to "{cls.data_path}"\n')


if __name__ == '__main__':
	PathManager._get_data_path(PathManager)
	PathManager.change_data_path(PathManager)
