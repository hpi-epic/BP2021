import os
from unittest.mock import mock_open, patch

import pytest

# import monitoring.exampleprinter as exampleprinter
import monitoring.svg_manipulation as svg_manipulation
# import tests.utils_tests as ut_t
from monitoring.svg_manipulation import SVGManipulator

svg_manipulator = SVGManipulator()


# def teardown_function(function):
# 	print('***TEARDOWN***')
# 	for f in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring'):
# 		if re.match('test_svg_*', f):
# 			os.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + f)


def test_get_default_dict():
	default_dict = svg_manipulation.get_default_dict()
	for key, val in default_dict.items():
		if key != 'simulation_name' and key != 'simulation_episode_length':
			assert val == ''


def test_replace_values():
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulator.save_overview_svg('test_svg_replace_values')
	assert 'the passed filename must end in .svg: ' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulator.write_dict_to_svg({'simulation_name': 0})
	assert 'the dictionary should only contain strings: ' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulator.save_overview_svg('test_svg_replace_values2.svg')
		svg_manipulator.save_overview_svg('test_svg_replace_values2.svg')
	assert 'the specified file already exists: ' in str(assertion_message.value)


correct_html = '<!doctype html>\n' + \
	'<html lang="de">\n' + \
	'	<head><meta charset="utf-8"/></head>\n' + \
	'	<img id="slideshow" src=""/>\n' + \
	'	<script>\n' + \
	'		images = [\n' + \
	'			{"name":"MarketOverview_001", "src":"./MarketOverview_001.svg"},\n' + \
	'			{"name":"MarketOverview_002", "src":"./MarketOverview_002.svg"},\n' + \
	'			{"name":"MarketOverview_003", "src":"./MarketOverview_003.svg"}\n' + \
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
	'		setInterval(changeImg, 1000)\n' + \
	'	</script>\n' + \
	'</html>\n'


def test_correct_html():
	global correct_html
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.isfile') as mock_isfile, \
		patch('monitoring.svg_manipulation.os.listdir') as mock_list_dir, \
		patch('builtins.open', mock_open()) as mock_file:
		mock_isfile.return_value = True
		mock_list_dir.return_value = ['MarketOverview_001.svg', 'MarketOverview_002.svg', 'MarketOverview_003.svg']

		svg_manipulator.to_html()

	mock_file.assert_called_once_with(os.path.join(svg_manipulator.save_directory, 'preview_svg.html'), 'w')
	handle = mock_file()
	handle.write.assert_called_once_with(correct_html)
