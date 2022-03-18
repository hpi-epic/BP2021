import argparse
import os
import shutil
from importlib import reload

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


def handle_command(command: str) -> None:
	"""
	Choose the file to run depending on the command given by the user.

	Default command is 'training'.

	Args:
		command (str): The command to perform.
	"""
	if command == 'training':
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
	parser = argparse.ArgumentParser(description='Customize your recommerce experience.')
	parser.add_argument('-d', '--datapath', type=str, help='Provide the path where `recommerce` will look for and save data')
	parser.add_argument('--get-defaults', action='store_true', help="""Default files, such as a `hyperparameter_config.json` and
trained models will be copied to your DATAPATH""")
	parser.add_argument('-u', '--unpack', action='store_true', help="""Use together with `--get-defaults`. Unpacks the default files so they are in
the correct relative locations to be used by the program.
NOTE: Any existing files with the same name as the default files will be overwritten!""")
	parser.add_argument('-c', '--command', type=str, choices=['training', 'exampleprinter', 'agent_monitoring'],
		default='training', help='The command to run')
	parser.add_argument('--no-action', action='store_true', help="""Set this flag if you do not want to run any command.
The default command is `training`""")

	args = parser.parse_args()

	# Handle provided arguments
	handle_datapath(args.datapath)
	# if default data was requested and should be unpacked, we can run the program
	if args.get_defaults and args.unpack:
		handle_unpack()
		if not args.no_action:
			handle_command(args.command)
	# if only default data was requested but not unpacked, the user probably doesn't have valid configuration files
	# so we don't start the command
	elif args.get_defaults:
		handle_get_defaults()
	elif args.unpack:
		raise argparse.ArgumentTypeError('The `--unpack` flag can only be used together with the `--get-defaults` flag')
	elif not args.no_action:
		handle_command(args.command)


if __name__ == '__main__':
	main()
