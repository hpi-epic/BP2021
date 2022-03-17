# used when installing the project as a pip package
import os

from setuptools import setup


# from: https://stackoverflow.com/questions/27664504/how-to-add-package-data-recursively-in-python-setup-py
def package_files(directory) -> list:
	"""
	This will make sure that ANY file within `recommerce/` is included in the pip package.

	Args:
		directory (str): The directory to check, in our case, 'recommerce'.

	Returns:
		list: The list of files found.
	"""
	paths = []
	for (path, directories, filenames) in os.walk(directory):
		paths.extend(os.path.join('..', path, filename) for filename in filenames)
	return paths


if __name__ == '__main__':
	extra_files = package_files('recommerce')
	setup(
		packages=['recommerce'],
		package_data={'': extra_files}
	)
