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
	return output_dict


class SVGManipulator():
	def __init__(self, save_dir: str = 'svg') -> None:
		self.value_dictionary = get_default_dict()
		# do not change the values in svg_template
		path_to_monitoring = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'monitoring'))
		with open(os.path.join(path_to_monitoring, 'MarketOverview_template.svg'), 'r') as template_svg:
			self.svg_template = template_svg.read()
		self.output_svg = None
		self.save_directory = os.path.join(path_to_monitoring, save_dir)

	def replace_one_value(self, target_key: str, value: str):
		"""
		Replace one value for a key in the dictionary.

		Args:
			target_key (str): a key in `self.value_dictionary`
			value (str): value for the provided key
		"""
		assert target_key in self.value_dictionary, 'Your specified key is not in the svg'
		assert isinstance(value, str), 'Please use strings as key values only'
		self.value_dictionary[target_key] = value
		self.write_dict_to_svg(target_dictionary=self.value_dictionary)

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

	def save_overview_svg(self, filename: str = 'MarketOverview_copy.svg') -> None:
		"""
		Save the stored svg data to a svg-file in BP2021/monitoring. If file already exists it will throw an error.

		Args:
			filename (str, optional): The target file name of the copy. Defaults to `MarketOverview_copy.svg`.
		"""
		assert filename.endswith('.svg'), f'the passed filename must end in .svg: {filename}'

		if not os.path.isdir(self.save_directory):
			os.mkdir(self.save_directory)

		file_path = os.path.join(self.save_directory, filename)
		assert not os.path.exists(file_path), f'the specified file already exists: {os.path.abspath(file_path)}'

		self.write_dict_to_svg(target_dictionary=self.value_dictionary)
		with open(file_path, 'w') as target_svg:
			target_svg.write(self.output_svg)

	def get_all_svg_from_directory(self, directory: str) -> list:
		"""
		Lists all svg files from a given directory and asserts, that they are all svgs.

		Args:
			Directory (str): Directory to get the svgs from

		Returns:
			list: List of svgs in this directory
		"""
		all_svg_files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
		assert all(file.endswith('.svg') for file in all_svg_files), 'All files in given directory (' + directory + ') must be svgs'
		return all_svg_files

	def to_gif(self, time : int = 500, gif_name: str = 'examplerun.gif') -> None:
		"""
		Converts all files in self.save_directory to one gif. All files in self.save_directory must be of type svg.
		If the number of files is large, this function will take a while.

		Args:
			time (int, optional): Time in ms for images to change. Defaults to 500.
		"""
		# this function is not tested, because it takes very long, we should discuss if we really need it
		all_svg_files = self.get_all_svg_from_directory(self.save_directory)

		# we need to convert svg to another format, because converting to gif does not work with svg
		all_drawings = [svg2rlg(os.path.join(self.save_directory, svg)) for svg in all_svg_files]
		for svg in all_drawings:
			svg.translate(0, -8420)
			svg.scale(5, 5)
		img, *imgs = [renderPM.drawToPIL(d) for d in all_drawings]

		# finally save it to gif
		gif_path = os.path.join(self.save_directory, gif_name)
		img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, duration=time, loop=0)
		print('You can find an animated overview at: ', gif_path)

	def construct_slideshow_html(self, images: list, time: int = 1000) -> str:
		"""
		returns the string to an html document with a slideshow of the given images on it

		Args:
			images (list): All images which should be in the slideshow
			time (int, optional): Duration of one image in the slideshow in ms. Defaults to 1000.

		Returns:
			str: The final full html document
		"""
		# slideshow view from: https://stackoverflow.com/questions/52478683/display-a-sequence-of-images-in-1-position-stored-in-an-object-js
		return '<!doctype html>\n' + \
			'<html lang="de">\n' + \
			'	<head><meta charset="utf-8"/></head>\n' + \
			'	<img id="slideshow" src="" style="width:100%"/>\n' + \
			'	<script>\n' + \
			'		images = [\n' + images + '\n'\
			'		];\n' + \
			'		imgIndex = 0;\n' + \
			'		function changeImg(){\n' + \
			'			document.getElementById("slideshow").src = images[imgIndex].src;\n' + \
			'				if(images.length > imgIndex+1){\n' + \
			'					imgIndex++;\n' + \
			'				} else {\n' + \
			'					imgIndex = 0;\n' + \
			'				}\n' + \
			'			}\n' + \
			'		changeImg();\n' + \
			'		setInterval(changeImg, ' + str(time) + ')\n' + \
			'	</script>\n' + \
			'</html>\n'

	def to_html(self, time: int = 1000, html_name: str = 'preview_svg.html') -> None:
		"""
		Writes an html document including a slideshow of all svgs in self.save_directory.

		Args:
			time (int, optional): Time in ms for images to change. Defaults to 1000.
			html_name (str, optional): Name for the html doument. Defaults to 'preview_svg.html'.
		"""
		assert html_name.endswith('.html'), f'the passed filename must end in .html: {html_name}'
		assert isinstance(time, int), 'time must be an int in ms'
		all_svgs = self.get_all_svg_from_directory(self.save_directory)

		# construct image array for js
		svg_array_for_js = ''
		for image in all_svgs:
			svg_array_for_js += '\t\t\t{"name":"' + image[:-4] + '", "src":"./' + image + '"},\n'

		# write html to file
		html_path = os.path.join(self.save_directory, html_name)
		with open(html_path, 'w') as out_file:
			out_file.write(self.construct_slideshow_html(svg_array_for_js[:-2], time))
		print('You can find an animated overview at: ', html_path)


def main():  # pragma: no cover
	"""
	This should be used for testing purposes only and is a way to quickly check if a configuration resulted in the correct `.svg`-output.
	"""
	manipulator = SVGManipulator()
	manipulator.write_dict_to_svg()
	manipulator.save_overview_svg()


if __name__ == '__main__':  # pragma: no cover
	main()
