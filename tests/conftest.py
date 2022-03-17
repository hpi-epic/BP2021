# Runs before the first test
import os
import shutil

from recommerce.configuration.path_manager import PathManager

# taken from https://stackoverflow.com/questions/17801300/how-to-run-a-method-before-all-tests-in-all-classes


def pytest_configure(config):
	"""
	Allows plugins and conftest files to perform initial configuration.
	This hook is called for every plugin and initial conftest
	file after command line options have been parsed.
	"""


def pytest_sessionstart(session):
	"""
	Called after the Session object has been created and
	before performing collection and entering the run test loop.
	"""
	PathManager.user_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data'))
	PathManager.data_path = PathManager.user_path
	PathManager.results_path = os.path.join(PathManager.user_path, os.pardir, 'test_results')


def pytest_sessionfinish(session, exitstatus):
	"""
	Called after whole test run finished, right before
	returning the exit status to the system.
	"""
	if os.path.exists(PathManager.results_path):
		shutil.rmtree(PathManager.results_path)


def pytest_unconfigure(config):
	"""
	called before test process is exited.
	"""
