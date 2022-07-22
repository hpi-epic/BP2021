import argparse
import os
import shutil
import sys
from importlib import metadata, reload

import recommerce.configuration.path_manager as path_manager


def handle_datapath(datapath: str) -> None:
	"""
	Update the datapath if requested.

	Args:
		datapath (str): The new path.
	"""
	path_manager.PathManager.manage_user_path(datapath)
	# reload to use the updated path
	reload(path_manager)


def handle_unpack():  # pragma: no cover only library calls
	"""
	Copy over the default files to the `user_path`, then unpack it in a way that everything is in the correct location.
	"""
	handle_get_defaults()
	shutil.copytree(os.path.join(path_manager.PathManager.user_path, 'default_data', 'configuration_files'),
		os.path.join(path_manager.PathManager.user_path, 'configuration_files'), dirs_exist_ok=True)
	shutil.copytree(os.path.join(path_manager.PathManager.user_path, 'default_data', 'modelfiles'),
		os.path.join(path_manager.PathManager.user_path, 'data'), dirs_exist_ok=True)
	shutil.rmtree(os.path.join(path_manager.PathManager.user_path, 'default_data'))
	print('The default data has been unpacked')


def handle_get_defaults() -> None:  # pragma: no cover only library calls
	"""
	Copy the contents of the `default_data` folder to the user_path.
	"""
	shutil.copytree(os.path.join(os.path.dirname(__file__), 'default_data'), os.path.join(path_manager.PathManager.user_path, 'default_data'),
		dirs_exist_ok=True)
	print(f'The default data was copied to your datapath at "{path_manager.PathManager.user_path}"')


def handle_command(command: str) -> None:
	"""
	If a command is given, choose the file to run.

	Args:
		command (str | None): The command to perform.
	"""
	if command is None:
		print('No --command provided, exiting...')
	elif command == 'training':
		from recommerce.rl import training_scenario
		training_scenario.main()
	elif command == 'exampleprinter':
		from recommerce.monitoring import exampleprinter
		exampleprinter.main()
	elif command == 'agent_monitoring':
		from recommerce.monitoring.agent_monitoring import am_monitoring
		am_monitoring.main()


def main():  # pragma: no cover
	"""
	The entrypoint for the `recommerce` application.

	Needs to be wrapped in a function to be callable as an entrypoint by the pip package.
	"""
	parser = argparse.ArgumentParser(description=f"""Train and Monitor Reinforcement-Learning Agents on various recommerce markets.
The current datapath is "{path_manager.PathManager.user_path}".""")

	parser.add_argument('-c', '--command', choices=('training', 'exampleprinter', 'agent_monitoring'),
		help='Choose and run the specified task.')

	parser.add_argument('-d', '--datapath', type=str, help="""Provide the path where `recommerce` will look for and save data.
Relative paths are supported.""")
	# We are cheesing a little bit here, since we always `handle_datapath` the datapath is printed anyways, no matter the command.
	# So this one is more to make the users happy and provide a command that doesn't do anything else
	parser.add_argument('--get-datapath', action='store_true', help='Will print the currently set datapath.')

	parser.add_argument('-gd', '--get-defaults', action='store_true', help="""default files, such as configuration files and
trained models will be copied to your `datapath`.""")
	parser.add_argument('-gdu', '--get-defaults-unpack', action='store_true', dest='unpack',
		help="""Works the same as --get-defaults, but also unpacks the default files so they are in the correct relative
locations to be used by `recommerce`. WARNING: Any existing files with the same name as the default files will be overwritten!""")

	parser.add_argument('-v', '--version', action='version', version=f'Recommerce Version {metadata.version("recommerce")}')

	args = parser.parse_args()

	if len(sys.argv) == 1:
		# display help message when no args are passed.
		print('Please provide an argument!')
		parser.print_help()

	# Handle provided arguments
	handle_datapath(args.datapath)

	if args.unpack:
		handle_unpack()
	elif args.get_defaults:
		handle_get_defaults()

	handle_command(args.command)


if __name__ == '__main__':
	main()
