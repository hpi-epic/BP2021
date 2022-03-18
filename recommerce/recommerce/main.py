import argparse
import os
import shutil
from importlib import metadata, reload

import coverage
import pytest

import recommerce.configuration.path_manager as path_manager


def handle_datapath(datapath: str) -> None:
	"""
	Update the datapath if requested.

	Args:
		datapath (str): The new path.
	"""
	path_manager.PathManager.manage_user_path(path_manager.PathManager, datapath)
	# reload to use the updated path
	reload(path_manager)


def handle_unpack():
	"""
	Copy over the default files to the `user_path`, then unpack it in a way that everything is in the correct location.
	"""
	handle_get_defaults()
	shutil.copytree(os.path.join(path_manager.PathManager.user_path, 'default_data', 'configuration_files'),
		path_manager.PathManager.user_path, dirs_exist_ok=True)
	shutil.copytree(os.path.join(path_manager.PathManager.user_path, 'default_data', 'modelfiles'),
		os.path.join(path_manager.PathManager.user_path, 'data'), dirs_exist_ok=True)
	shutil.rmtree(os.path.join(path_manager.PathManager.user_path, 'default_data'))
	print('The default data has been unpacked')


def handle_get_defaults() -> None:
	"""
	Copy the contents of the `default_data` folder to the user_path.
	"""
	shutil.copytree(os.path.join(os.path.dirname(__file__), 'default_data'), os.path.join(path_manager.PathManager.user_path, 'default_data'),
		dirs_exist_ok=True)
	print(f'The default data was copied to your datapath at "{path_manager.PathManager.user_path}"')


def handle_tests() -> None:
	"""
	Run the test suite located in the datapath while tracking coverage. Also writes a coverage.svg to the datapath
	"""
	print('Running tests...')
	cov = coverage.Coverage()
	cov.start()

	pytest.main([])

	cov.stop()
	cov.save()

	cov.json_report()

	# coverage-badge unfortunately does not provide an interface
	os.makedirs(os.path.join(path_manager.PathManager.user_path, os.pardir, os.pardir, 'badges'), exist_ok=True)
	os.system(f'coverage-badge -f -o "{os.path.join(path_manager.PathManager.user_path, os.pardir, os.pardir, "badges", "coverage.svg")}"')

	cov.report()


def handle_command(command: str) -> None:
	"""
	If a command is given, choose the file to run.

	Args:
		command (str | None): The command to perform.
	"""
	if command is None:
		return
	elif command == 'training':
		from recommerce.rl import training_scenario
		training_scenario.main()
	elif command == 'exampleprinter':
		from recommerce.monitoring import exampleprinter
		exampleprinter.main()
	elif command == 'agent_monitoring':
		from recommerce.monitoring.agent_monitoring import am_monitoring
		am_monitoring.main()


def main():
	"""
	The entrypoint for the `recommerce` application.

	Needs to be wrapped in a function to be callable as an entrypoint by the pip package.
	"""
	parser = argparse.ArgumentParser(description='Train and Monitor Reinforcement-Learning Agents on various Circular-Economy models.')

	parser.add_argument('-c', '--command', choices=('training', 'exampleprinter', 'agent_monitoring'),
		help='choose the command to run')

	parser.add_argument('-d', '--datapath', type=str, help="""provide the path where `recommerce` will look for and save data.
Relative paths are supported""")

	parser.add_argument('--get-defaults', action='store_true', help="""default files, such as a hyperparameter_config.json and
trained models will be copied to your DATAPATH""")
	parser.add_argument('--get-defaults-unpack', action='store_true', dest='unpack',
		help="""Works the same as --get-defaults, but also unpacks the default files so they are in the correct relative
locations to be used by the program. Has priority over --get-defaults.
NOTE: Any existing files with the same name as the default files will be overwritten!""")

	parser.add_argument('-t', '--test', action='store_true', help="""run pytest on tests stored in the datapath.
Also tracks coverage and creates a coverage badge in datapath/badges/coverage.svg. Has priority over --command""")

	parser.add_argument('-v', '--version', action='version', version=f'Recommerce Version {metadata.version("recommerce")}')

	args = parser.parse_args()

	# Handle provided arguments
	handle_datapath(args.datapath)

	if args.unpack:
		handle_unpack()
	elif args.get_defaults:
		handle_get_defaults()

	if args.test:
		handle_tests()
	else:
		handle_command(args.command)


if __name__ == '__main__':
	main()
