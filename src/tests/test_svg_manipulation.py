import os
from unittest.mock import mock_open, patch

import pytest

import monitoring.exampleprinter as exampleprinter
import monitoring.svg_manipulation as svg_manipulation
import tests.utils_tests as ut_t

svg_manipulator = svg_manipulation.SVGManipulator()


def setup_function(function):
	print('***SETUP***')
	global svg_manipulator
	svg_manipulator = svg_manipulation.SVGManipulator()


def test_get_default_dict():
	default_dict = svg_manipulation.get_default_dict()
	for key, val in default_dict.items():
		assert val == ''


def test_replace_one_value():
	global svg_manipulator
	bevor_dict = svg_manipulator.value_dictionary
	assert '' == bevor_dict['simulation_name']
	svg_manipulator.replace_one_value('simulation_name', 'new_name')
	assert 'new_name' == svg_manipulator.value_dictionary['simulation_name']


def test_write_dict_to_svg():
	global svg_manipulator
	test_dict = svg_manipulation.get_default_dict()
	for key in test_dict:
		test_dict[key] = 'test'
	svg_manipulator.write_dict_to_svg(test_dict)
	correct_svg = ''
	with open(os.path.join(os.path.dirname(__file__), 'output_test_svg.svg')) as file:
		correct_svg = file.read()
	assert correct_svg == svg_manipulator.output_svg


# test save_overview_svg
def test_file_name_for_save_ends_with_svg():
	global svg_manipulator
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulator.save_overview_svg('test_svg_replace_values')
	assert 'the passed filename must end in .svg: ' in str(assertion_message.value)


def test_file_file_should_not_exist():
	global svg_manipulator
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
		patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir, \
		patch('builtins.open', mock_open()):
		mock_isdir.return_value = True
		mock_exists.return_value = False

		with pytest.raises(AssertionError) as assertion_message:
			svg_manipulator.save_overview_svg('test_svg_replace_values2.svg')
			mock_exists.return_value = True
			svg_manipulator.save_overview_svg('test_svg_replace_values2.svg')
		assert 'the specified file already exists: ' in str(assertion_message.value)


def test_write_to_dict_only_strings():
	global svg_manipulator
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
		patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir:
		mock_isdir.return_value = True
		mock_exists.return_value = False

		with pytest.raises(AssertionError) as assertion_message:
			svg_manipulator.write_dict_to_svg({'simulation_name': 0})
		assert 'the dictionary should only contain strings: ' in str(assertion_message.value)


def test_replace_values():
	global svg_manipulator
	svg_manipulator.output_svg = 'Hello World!'
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir, \
		patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
		patch('monitoring.svg_manipulation.SVGManipulator.write_dict_to_svg'), \
		patch('builtins.open', mock_open()) as mock_file:
		mock_isdir.return_value = True
		mock_exists.return_value = False

		# run saving process
		svg_manipulator.save_overview_svg('my_test_file.svg')

	# assert that file would exsist and the content would be the wanted content
	mock_file.assert_called_once_with(os.path.join(svg_manipulator.save_directory, 'my_test_file.svg'), 'w')
	mock_file().write.assert_called_once_with('Hello World!')


correct_html = '<!doctype html>\n' + \
	'<html lang="de">\n' + \
	'	<head><meta charset="utf-8"/></head>\n' + \
	'	<img id="slideshow" src="" style="width:100%"/>\n' + \
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
	global svg_manipulator
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.isfile') as mock_isfile, \
		patch('monitoring.svg_manipulation.os.listdir') as mock_list_dir, \
		patch('builtins.open', mock_open()) as mock_file:
		mock_isfile.return_value = True
		mock_list_dir.return_value = ['MarketOverview_001.svg', 'MarketOverview_002.svg', 'MarketOverview_003.svg']

		# run the convertion to html
		svg_manipulator.to_html()

	# assert that file would exsist and the content would be correct
	mock_file.assert_called_once_with(os.path.join(svg_manipulator.save_directory, 'preview_svg.html'), 'w')
	mock_file().write.assert_called_once_with(correct_html)


def test_one_exampleprinter_run():
	global correct_html

	# use only three episodes for reusing the correct_html
	json = ut_t.create_mock_json_sim_market(episode_size='3')
	with patch('builtins.open', mock_open(read_data=json)) as utils_mock_file:
		ut_t.check_mock_file_sim_market(utils_mock_file, json)
		# initialize all functions to be mocked
		with patch('monitoring.exampleprinter.ut.write_dict_to_tensorboard'), \
			patch('monitoring.svg_manipulation.os.path.isfile') as mock_isfile, \
			patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir, \
			patch('monitoring.svg_manipulation.os.listdir') as mock_list_dir, \
			patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
			patch('builtins.open', mock_open()) as mock_file:
			mock_isfile.return_value = True
			mock_isdir.return_value = True
			mock_exists.return_value = False
			mock_list_dir.return_value = ['MarketOverview_001.svg', 'MarketOverview_002.svg', 'MarketOverview_003.svg']

			exampleprinter.run_example()
		# asserts that the html has been written
		mock_file().write.assert_called_with(correct_html)
