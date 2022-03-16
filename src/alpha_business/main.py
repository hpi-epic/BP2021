import argparse

from alpha_business.configuration.path_manager import PathManager
from alpha_business.monitoring import exampleprinter
from alpha_business.monitoring.agent_monitoring import am_monitoring
from alpha_business.rl import training_scenario

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Customize your alpha_business experience.')
	parser.add_argument('--datapath', type=str, help='Provide the path where alpha_business will look for and save data')
	parser.add_argument('--get-defaults', action='store_true', help="""Default files, such as a `hyperparameter_config.json` and
trained models will be saved to your `data_path`""")
	parser.add_argument('-c', '--command', type=str, choices=['training', 'exampleprinter', 'agent_monitoring'],
		default='training', help='Choose the command to run')

	args = parser.parse_args()

	# --datapath
	# Update the datapath if possible
	PathManager.manage_user_path(PathManager, args.datapath)

	# --get-defaults
	# Copy the contents of `./src/alpha_business/default_data` to the user provided data path
	if args.get_defaults:
		# DO MAGIC HERE
		# Any default files should be copied over to the data_path directory
		pass

	# --command
	# Choose the file to run depending on the command
	if args.command == 'training':
		training_scenario.main()
	if args.command == 'exampleprinter':
		exampleprinter.main()
	if args.command == 'agent_monitoring':
		am_monitoring.main()
