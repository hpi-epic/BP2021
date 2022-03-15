import argparse
import os

from alpha_business.configuration.path_manager import PathManager
from alpha_business.monitoring import exampleprinter
from alpha_business.monitoring.agent_monitoring import am_monitoring
from alpha_business.rl import training_scenario


def readable_dir(prospective_dir) -> str:
	"""
	Helper function defining whether or not a prospective_dir is an existing and readable path.

	Adapted from https://stackoverflow.com/questions/11415570/directory-path-types-with-argparse

	Args:
		prospective_dir (str): The path to check.

	Returns:
		str: The path.
	"""
	if not os.path.isdir(prospective_dir):
		raise argparse.ArgumentTypeError(f'The provided directory does not exist: {prospective_dir}')
	if os.access(prospective_dir, os.R_OK):
		return prospective_dir
	else:
		raise argparse.ArgumentTypeError(f'The provided directory is not readable: {prospective_dir}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Customize your alpha_business experience.')
	parser.add_argument('-c', '--command', type=str, choices=['training', 'exampleprinter', 'agent_monitoring'],
		default='training', help='Choose the command to run')
	parser.add_argument('--datapath', type=readable_dir, help='Provide the path where alpha_business will look for and save data')

	args = parser.parse_args()

	# Save the new datapath
	if args.datapath is not None:
		PathManager.update_data_path(PathManager, args.datapath)

	# Choose the file to run depending on the command
	if args.command == 'training':
		training_scenario.main()
	if args.command == 'exampleprinter':
		exampleprinter.main()
	if args.command == 'agent_monitoring':
		am_monitoring.main()
