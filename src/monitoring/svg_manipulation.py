import os

from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg


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
		'a_garbage',
		'a_inventory',
		'a_profit',
		'a_throw_away',
		'a_price_new',
		'a_price_used',
		'a_rebuy_price',
		'a_repurchases',
		'a_resource_cost',
		'a_resources_in_use',
		'a_sales_new',
		'a_sales_used',
		'b_competitor_name',
		'b_inventory',
		'b_profit',
		'b_price_new',
		'b_price_used',
		'b_rebuy_price',
		'b_repurchases',
		'b_resource_cost',
		'b_sales_new',
		'b_sales_used'
	]
	output_dict = dict.fromkeys(keys, '')
	output_dict['simulation_name'] = 'Market Simulation'
	# output_dict['simulation_episode_length'] = str(ut.EPISODE_LENGTH)
	return output_dict


class SVGManipulator():
	def __init__(self, save_directory='svg') -> None:
		self.value_dictionary = get_default_dict()
		# do not change the values in svg_template
		with open('./monitoring/MarketOverview_template.svg', 'r') as template_svg:
			self.svg_template = template_svg.read()
		self.output_svg = None
		self.save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + save_directory

	def replace_one_value(self, target_key, value):
		"""
		Replace one value for a key in the dictionary.

		Args:
			target_key (str): a key in `self.value_dictionary`
			value (str): value for the provided key
		"""
		assert target_key in self.value_dictionary
		assert isinstance(value, str)
		self.value_dictionary[target_key] = value
		self.write_dict_to_svg(target_dictionary=self.value_dictionary)

	def save_overview_svg(self, filename: str = 'MarketOverview_copy.svg') -> None:
		"""
		Save the stored svg data to a svg-file in BP2021/monitoring. If file already exists it will throw an error.

		Args:
			filename (str, optional): The target file name of the copy. Defaults to `MarketOverview_copy.svg`.
		"""
		assert filename.endswith('.svg'), f'the passed filename must end in .svg: {filename}'
		if not os.path.isdir(self.save_dir):
			os.mkdir(self.save_dir)
			print(self.save_dir)
		filename = self.save_dir + os.sep + filename
		assert not os.path.exists(filename), f'the specified file already exists: {os.path.abspath(filename)}'

		self.write_dict_to_svg(target_dictionary=self.value_dictionary)
		with open(filename, 'w') as target_svg:
			target_svg.write(self.output_svg)

	def write_dict_to_svg(self, target_dictionary: dict = get_default_dict()) -> str:
		"""
		Replace all placeholder values in the current svg with a given dictionary.

		Args:
			target_dictionary (dict, optional): Dictionary containing the values that should be replaced in the copy. Defaults to `get_default_dict()`.

		Returns:
			str: The full path to the copied file.
		"""
		assert all(isinstance(value, str) for _, value in target_dictionary.items()), f'the dictionary should only contain strings: {target_dictionary}'

		# reset the output svg to the template to be able to replace the placeholders by actual values
		self.output_svg = self.svg_template
		for key, value in target_dictionary.items():
			self.output_svg = self.output_svg.replace(key, value)

	def convert_svg_sequence_to_gif(self):
		onlyfiles = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
		img, *imgs = [renderPM.drawToPIL(svg2rlg(os.path.join(self.save_dir, f))) for f in onlyfiles]
		img.save(fp=os.path.join(self.save_dir, 'examplerun.gif'), format='GIF', append_images=imgs, save_all=True, duration=500, loop=0)


def main():  # pragma: no cover
	"""
	This should be used for testing purposes only and is a way to quickly check if a configuration resulted in the correct `.svg`-output.
	"""
	manipulator = SVGManipulator()
	manipulator.write_dict_to_svg()
	manipulator.save_overview_svg()


if __name__ == '__main__':  # pragma: no cover
	main()
