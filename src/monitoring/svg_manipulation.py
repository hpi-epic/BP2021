import shutil

import configuration.utils_sim_market as ut


def main():
	src = './monitoring/MarketOverview_template.svg'
	dst = './monitoring/MarketOverview_copy.svg'

	# Copy the template file to dst
	shutil.copyfile(src, dst)

	# open the copied file
	copied_file = open('./monitoring/MarketOverview_copy.svg', 'rt')
	# read the data
	# returns a string
	data = copied_file.read()
	# replace all occurances of the first string with the second string
	# each variable that should be substituted has been given an appropiate placeholder name in the template
	data = data.replace('simulation_episode_length', str(ut.EPISODE_LENGTH))
	# close and re-open the file
	# we need to do this because you cannot open a file with read and write permissions (wrt)
	copied_file.close()
	copied_file = open('./monitoring/MarketOverview_copy.svg', 'wt')
	# write the modified svg data back to the target file
	copied_file.write(data)
	copied_file.close()


if __name__ == '__main__':  # pragma: no cover
	main()
