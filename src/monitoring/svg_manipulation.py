import os

import configuration.utils_sim_market as ut


def get_default_dict() -> dict:
	"""
	Return a default dictionary to be used with `replace_values()`.

	Contains some pre-filled information from e.g. utils_sim_market.py.
	Non-default values will be left empty.

	Returns:
		dict: The default dictionary.
	"""
	keys = [
		'simulation_name',
		'simulation_episode_length',
		'simulation_current_episode',
		'consumer_total_arrivals',
		'consumer_total_sales',
		'a_competitor_name',
		'a_resource_cost',
		'a_resource_purchases',
		'a_price_new',
		'a_sales_new',
		'a_price_used',
		'a_sales_used',
		'a_rebuy_price',
		'a_repurchases',
		'a_resources_in_use',
		'a_garbage',
		'b_competitor_name',
		'b_resource_cost',
		'b_resource_purchases',
		'b_price_new',
		'b_sales_new',
		'b_price_used',
		'b_sales_used',
		'b_rebuy_price',
		'b_repurchases',
		'b_resources_in_use',
		'b_garbage'
	]
	output_dict = dict.fromkeys(keys, '')
	output_dict['simulation_name'] = 'Market Simulation'
	output_dict['simulation_episode_length'] = str(ut.EPISODE_LENGTH)
	return output_dict


def replace_values(filename: str = './monitoring/MarketOverview_copy.svg', target_dictionary: dict = get_default_dict()) -> str:
	"""
	Create a copy of the template `MarketOverview.svg` file and replace the placeholder values with the given ones.

	Args:
		filename (str, optional): The target file name of the copy. Defaults to `./monitoring/MarketOverview_copy.svg`.
		target_dictionary (dict, optional): Dictionary containing the values that should be replaced in the copy. Defaults to `get_default_dict()`.

	Returns:
		str: The full path to the copied file.
	"""
	assert all(isinstance(value, str) for key, value in target_dictionary.items()), f'the dictionary should only contain strings: {target_dictionary}'
	assert not os.path.exists(filename), f'the specified file already exists: {os.path.abspath(filename)}'

	template_file = open('./monitoring/MarketOverview_template.svg', 'r')
	data = template_file.read()
	template_file.close()

	for key, value in target_dictionary.items():
		data = data.replace(key, value)

	target_file = open(filename, 'w')
	target_file.write(data)
	target_file.close()
	return os.path.abspath(filename)


def main():
	"""
	This should be used for testing purposes only and is a way to quickly check if a configuration resulted in the correct `.svg`-output.
	"""
	get_default_dict()
	replace_values()


if __name__ == '__main__':  # pragma: no cover
	main()
