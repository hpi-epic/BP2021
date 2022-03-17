# used when installing the project as a pip package
from setuptools import setup

if __name__ == '__main__':
	# package data defines folders with non-.py files which should also be included in the pip package
	setup(package_data={
		'alpha_business': ['default_data/*/*',
			'monitoring/data/MarketOverview_template.svg',
			'configuration/user_path.txt']
	})
