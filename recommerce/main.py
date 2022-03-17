import argparse
import os
import shutil
from importlib import reload

import recommerce.configuration.path_manager as path_manager


def main():
	"""
	The entrypoint for the `recommerce` application.

	Needs to be wrapped in a function to be callable as an entrypoint by the pip package.
	"""
	parser = argparse.ArgumentParser(description='Customize your recommerce experience.')
	parser.add_argument('--datapath', type=str, help='Provide the path where `recommerce` will look for and save data')
	parser.add_argument('--get-defaults', action='store_true', help="""Default files, such as a `hyperparameter_config.json` and
trained models will be copied to your `data_path`""")
	parser.add_argument('-c', '--command', type=str, choices=['training', 'exampleprinter', 'agent_monitoring'],
		default='training', help='Choose the command to run')

	args = parser.parse_args()

	# --datapath
	# Update the datapath if possible
	path_manager.PathManager.manage_user_path(path_manager.PathManager, args.datapath)
	# reload to use the updated path
	reload(path_manager)

	# --get-defaults
	# Copy the contents of `./src/recommerce/default_data` to the user provided data path
	if args.get_defaults:
		shutil.copytree(os.path.join(os.path.dirname(__file__), 'default_data'), os.path.join(path_manager.PathManager.user_path, 'default_data'),
			dirs_exist_ok=True)

	# --command
	# Choose the file to run depending on the command
	if args.command == 'training':
		from recommerce.rl import training_scenario
		training_scenario.main()
	if args.command == 'exampleprinter':
		from recommerce.monitoring import exampleprinter
		exampleprinter.main()
	if args.command == 'agent_monitoring':
		from recommerce.monitoring.agent_monitoring import am_monitoring
		am_monitoring.main()


if __name__ == '__main__':
	main()
