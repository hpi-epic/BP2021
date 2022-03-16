# used when installing the project as a pip package
from setuptools import setup

if __name__ == '__main__':
	# package data defines folders with non-.py files which should also be included in the pip package
	# in our case, this is the default data we want to provide the user with
	setup(package_data={
		'alpha_business': ['default_data/*/*']
	})
