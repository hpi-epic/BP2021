import argparse

from alpha_business.configuration.path_manager import PathManager
from alpha_business.monitoring import exampleprinter
from alpha_business.monitoring.agent_monitoring import am_monitoring
from alpha_business.rl import training_scenario

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Customize your alpha_business experience.')
	parser.add_argument('-c', '--command', type=str, choices=['training', 'exampleprinter', 'agent_monitoring'],
		default='training', help='Choose the command to run')
	parser.add_argument('--datapath', type=str, help='Provide the path where alpha_business will look for and save data')

	args = parser.parse_args()

	# Manage the datapath
	PathManager.manage_data_path(PathManager, args.datapath)

	# Choose the file to run depending on the command
	if args.command == 'training':
		training_scenario.main()
	if args.command == 'exampleprinter':
		exampleprinter.main()
	if args.command == 'agent_monitoring':
		am_monitoring.main()
