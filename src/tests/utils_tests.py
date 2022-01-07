import os


def create_mock_json_rl(gamma='0.99', batch_size='32', replay_size='100000', learning_rate='1e-6', sync_target_frames='1000', replay_start_size='10000', epsilon_decay_last_frame='75000', epsilon_start='1.0', epsilon_final='0.1') -> str:
	"""
	Create a string in JSON format that can be used to mock the config_rl.json file.

	Args:
		gamma (str, optional): Defaults to '0.99'.
		batch_size (str, optional): Defaults to '32'.
		replay_size (str, optional): Defaults to '100000'.
		learning_rate (str, optional): Defaults to '1e-6'.
		sync_target_frames (str, optional): Defaults to '1000'.
		replay_start_size (str, optional): Defaults to '10000'.
		epsilon_decay_last_frame (str, optional): Defaults to '75000'.
		epsilon_start (str, optional): Defaults to '1.0'.
		epsilon_final (str, optional): Defaults to '0.1'.

	Returns:
		str: A string in JSON format.
	"""
	return '{\n\t"gamma" : ' + gamma + ',\n' + \
		'\t"batch_size" : ' + batch_size + ',\n' + \
		'\t"replay_size" : ' + replay_size + ',\n' + \
		'\t"learning_rate" : ' + learning_rate + ',\n' + \
		'\t"sync_target_frames" : ' + sync_target_frames + ',\n' + \
		'\t"replay_start_size" : ' + replay_start_size + ',\n' + \
		'\t"epsilon_decay_last_frame" : ' + epsilon_decay_last_frame + ',\n' + \
		'\t"epsilon_start" : ' + epsilon_start + ',\n' + \
		'\t"epsilon_final" : ' + epsilon_final + '\n' + \
		'}'


def check_mock_file_rl(mock_file, json) -> None:
	"""
	Confirm that a mock JSON for the config_rl.json is read correctly.

	Args:
		mock_file (unittest.mock.MagicMock): The mocked file.
		json (str): The mock JSON string to be checked.
	"""
	path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'config_rl.json')
	assert open(path).read() == json, 'the mock did not work correctly, as the read file was not equal to the set mock-json'
	mock_file.assert_called_with(path)


def create_mock_json_sim_market(episode_size='20', max_price='15', max_quality='100', number_of_customers='30', production_price='5') -> str:
	"""
	Create a string in JSON format that can be used to mock the config_sim_market.json file.

	Args:
		episode_size (str, optional): Defaults to '20'.
		max_price (str, optional): Defaults to '15'.
		max_quality (str, optional): Defaults to '100'.
		number_of_customers (str, optional): Defaults to '30'.
		production_price (str, optional): Defaults to '5'.

	Returns:
		str: A string in JSON format.
	"""
	return '{\n\t"episode_size": ' + episode_size + ',\n' + \
		'\t"max_price": ' + max_price + ',\n' + \
		'\t"max_quality": ' + max_quality + ',\n' + \
		'\t"number_of_customers": ' + number_of_customers + ',\n' + \
		'\t"production_price": ' + production_price + '\n}'


def check_mock_file_sim_market(mock_file, json) -> None:
	"""
	Confirm that a mock JSON for the config_sim_market.json is read correctly.

	Args:
		mock_file (unittest.mock.MagicMock): The mocked file.
		json (str): The mock JSON string to be checked.
	"""
	path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'config_sim_market.json')
	assert open(path).read() == json, 'the mock did not work correctly, as the read file was not equal to the set mock-json'
	mock_file.assert_called_with(path)


def remove_line(number, json) -> str:
	"""
	Remove the specified line from a mock JSON string.

	Args:
		number (int): The line that should be removed.
		json (str): The JSON string from which to remove the line.

	Returns:
		str: The JSON string with the missing line.
	"""
	lines = json.split('\n')
	final_lines = lines[:number + 1]
	final_lines += lines[number + 2:len(lines)]
	final_lines[-2] = final_lines[-2].replace(',', '')
	return '\n'.join(final_lines)


def create_mock_rewards(num_entries) -> list:
	"""
	Create a list of ints to be used as e.g. mock rewards.

	Args:
		num_entries (int): How many numbers should be in the list going from 1 to num_entries.

	Returns:
		list: The list of rewards.
	"""
	return list(range(1, num_entries))
