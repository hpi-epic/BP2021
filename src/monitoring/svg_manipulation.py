import os


def get_default_dict() -> dict:
	"""
	Return a default dictionary to be used with `replace_values()`.

	Contains some pre-filled information.
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
	# output_dict['simulation_episode_length'] = str(ut.EPISODE_LENGTH)
	return output_dict


class SVGManipulator():
	def __init__(self) -> None:
		self.value_dictionary = get_default_dict()
		with open('./monitoring/MarketOverview_template.svg', 'r') as template_svg:
			self.svg_data = template_svg.read()

	def replace_one_value(self, target_key, value):
		"""
		Replaces one value for a key in the dictionary

		Args:
			target_key (str): a key in svg dictionary
			value (str): value for the provided key
		"""
		assert target_key in self.value_dictionary
		assert isinstance(value, str)
		self.value_dictionary[target_key] = value
		self.replace_all_values_with_dict(self, target_dictionary=self.value_dictionary)

	def save_overview_svg(self, filename: str = 'MarketOverview_copy.svg') -> None:
		"""
		Save the stored svg data to a svg-file in BP2021/monitoring. If file already exists it will throw an error

		Args:
			filename (str, optional): The target file name of the copy. Defaults to `MarketOverview_copy.svg`.
		"""
		assert filename.endswith('.svg'), f'the passed filename must end in .svg: {filename}'
		filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + filename
		assert not os.path.exists(filename), f'the specified file already exists: {os.path.abspath(filename)}'

		with open(filename, 'w') as target_svg:
			target_svg.write(self.svg_data)

	def replace_all_values_with_dict(self, filename: str = 'MarketOverview_copy.svg', target_dictionary: dict = get_default_dict()) -> str:
		"""
		Replaces all values in the current svg with a given dictionary

		Args:
			target_dictionary (dict, optional): Dictionary containing the values that should be replaced in the copy. Defaults to `get_default_dict()`.

		Returns:
			str: The full path to the copied file.
		"""
		assert all(isinstance(value, str) for _, value in target_dictionary.items()), f'the dictionary should only contain strings: {target_dictionary}'

		for key, value in target_dictionary.items():
			self.svg_data = self.svg_data.replace(key, value)


def main():  # pragma: no cover
	"""
	This should be used for testing purposes only and is a way to quickly check if a configuration resulted in the correct `.svg`-output.
	"""
	manipulator = SVGManipulator()
	get_default_dict()
	manipulator.replace_all_values_with_dict()


if __name__ == '__main__':  # pragma: no cover
	main()
