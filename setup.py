# used when installing the project as a pip package
import os

from setuptools import setup


def package_files(directory):
	paths = []
	for (path, directories, filenames) in os.walk(directory):
		paths.extend(os.path.join('..', path, filename) for filename in filenames)
	return paths


extra_files = package_files('alpha_business')

if __name__ == '__main__':
	# package data defines folders with non-.py files which should also be included in the pip package
	setup(
		packages=['alpha_business'],
		package_data={'': extra_files, 'alpha_business': ['default_data/*/*']}
	)
