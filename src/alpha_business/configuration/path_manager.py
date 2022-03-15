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
			cls.data_path = path_file.read()

		# There is a valid path saved
		if os.path.isdir(cls.data_path):
			print(f'Data will be read from and saved to "{cls.data_path}"')
			return
		elif cls.data_path != '':
			print(f'The current saved data path does not exist: {cls.data_path}')

		# No path provided as of yet
		while not os.path.isdir(cls.data_path):
			cls.data_path = input('Please provide a path where alpha_business will look for and save data:\n')
			if not os.path.isdir(cls.data_path):
				print(f'The provided path is not a valid directory: {cls.data_path}')

		with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'w') as path_file:
			path_file.write(cls.data_path)

		print(f'Data will be read from and saved to "{cls.data_path}"\n')

	def change_data_path(cls):
		"""
		Use this to manually update the data path, even when it is valid.
		"""
		new_data_path = ''

		while not os.path.isdir(new_data_path):
			new_data_path = input('Please provide a path where alpha_business will look for and save data, or "exit" to not change anything:\n')
			if new_data_path == 'exit':
				print(f'Data will be read from and saved to "{cls.data_path}"')
				return
			if not os.path.isdir(new_data_path):
				print(f'The provided path is not a valid directory: {new_data_path}')

		cls.data_path = new_data_path
		with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'w') as path_file:
			path_file.write(cls.data_path)

		print(f'Data will be read from and saved to "{cls.data_path}"\n')


if __name__ == '__main__':
	PathManager._get_data_path(PathManager)
	PathManager.change_data_path(PathManager)
