import os
from unittest.mock import mock_open, patch

import pytest

import monitoring.svg_manipulation as svg_manipulation
import tests.utils_tests as ut_t
from monitoring.exampleprinter import ExamplePrinter

svg_manipulator = svg_manipulation.SVGManipulator()


def setup_function(function):
	print('***SETUP***')
	global svg_manipulator
	svg_manipulator = svg_manipulation.SVGManipulator()


def test_get_default_dict():
	default_dict = svg_manipulation.get_default_dict()
	for _, val in default_dict.items():
		assert val == ''


def test_correct_template():
	with open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'monitoring', 'MarketOverview_template.svg')), 'r') as template:
		correct_template = template.read()
	assert correct_template == svg_manipulator.template_svg

	# run one exampleprinter and to make sure the template does not get changed
	json = ut_t.create_mock_json_sim_market(episode_size='3')
	with patch('builtins.open', mock_open(read_data=json)) as utils_mock_file:
		ut_t.check_mock_file(utils_mock_file, json)
		# initialize all functions to be mocked
		with patch('monitoring.exampleprinter.ut.write_dict_to_tensorboard'), \
			patch('monitoring.svg_manipulation.os.path.isfile') as mock_isfile, \
			patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir, \
			patch('monitoring.svg_manipulation.os.listdir') as mock_list_dir, \
			patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
			patch('builtins.open', mock_open()):
			mock_isfile.return_value = True
			mock_isdir.return_value = True
			mock_exists.return_value = False
			mock_list_dir.return_value = ['MarketOverview_001.svg', 'MarketOverview_002.svg', 'MarketOverview_003.svg']

			ExamplePrinter().run_example()
		assert correct_template == svg_manipulator.template_svg


def test_replace_one_value():
	assert '' == svg_manipulator.value_dictionary['simulation_name']
	svg_manipulator.replace_one_value('simulation_name', 'new_name')
	assert 'new_name' == svg_manipulator.value_dictionary['simulation_name']


def test_key_not_in_dict():
	with pytest.raises(AssertionError) as assertion_info:
		svg_manipulator.replace_one_value('not_in_dict', 'none')
	assert 'the provided key does not exist in the dictionary:' in str(assertion_info.value)


def test_value_not_string():
	with pytest.raises(AssertionError) as assertion_info:
		svg_manipulator.replace_one_value('simulation_name', 1)
	assert 'the provided value must be of type str but was' in str(assertion_info.value)


def test_write_dict_to_svg():
	test_dict = svg_manipulation.get_default_dict()
	for key in test_dict:
		test_dict[key] = 'test'
	svg_manipulator.write_dict_to_svg(test_dict)
	correct_svg = ''
	with open(os.path.join(os.path.dirname(__file__), 'output_test_svg.svg')) as file:
		correct_svg = file.read()
	assert correct_svg == svg_manipulator.output_svg
	assert test_dict == svg_manipulator.value_dictionary


# tests below test save_overview_svg()
def test_file_should_not_exist():
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
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
		patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir:
		mock_isdir.return_value = True
		mock_exists.return_value = False

		with pytest.raises(AssertionError) as assertion_message:
			svg_manipulator.write_dict_to_svg({'simulation_name': 0})
		assert 'the dictionary should only contain strings: ' in str(assertion_message.value)


def test_replace_values():
	svg_manipulator.output_svg = 'Hello World!'
	# initialize all functions to be mocked
	with patch('monitoring.svg_manipulation.os.path.isdir') as mock_isdir, \
		patch('monitoring.svg_manipulation.os.path.exists') as mock_exists, \
		patch('monitoring.svg_manipulation.SVGManipulator.write_dict_to_svg'), \
		patch('builtins.open', mock_open()) as mock_file:
		mock_isdir.return_value = True
		mock_exists.return_value = False

		# run saving process
		svg_manipulator.save_overview_svg('my_test_file')

	# assert that file would exsist and the content would be the wanted content
	mock_file.assert_called_once_with(os.path.join(svg_manipulator.save_directory, 'my_test_file.svg'), 'w')
	mock_file().write.assert_called_once_with('Hello World!')


def test_files_are_svgs():
	files_in_dir = ['MarketOverview_001.svg', 'MarketOverview_002.svg', 'MarketOverview_003.svg']
	with patch('monitoring.svg_manipulation.os.path.isfile') as mock_isfile, \
		patch('monitoring.svg_manipulation.os.listdir') as mock_list_dir:
		mock_isfile.return_value = True
		mock_list_dir.return_value = files_in_dir

		listed_files = svg_manipulator.get_all_svg_from_directory('./test_dir')
		assert files_in_dir == listed_files

		files_in_dir += 'test.png'

		with pytest.raises(AssertionError) as assertion_info:
			svg_manipulator.get_all_svg_from_directory('./test_dir')
		assert 'all files in given directory must be svgs:' in str(assertion_info.value)


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


def test_time_not_int():
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulator.to_html(time='test_svg_replace_values')
	assert 'time must be an int in ms' in str(assertion_message.value)


def test_one_exampleprinter_run():
	# use only three episodes for reusing the correct_html
	json = ut_t.create_mock_json_sim_market(episode_size='3')
	with patch('builtins.open', mock_open(read_data=json)) as utils_mock_file:
		ut_t.check_mock_file(utils_mock_file, json)
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

			ExamplePrinter().run_example()
		# asserts that the html has been written
		mock_file().write.assert_called_with(correct_html)
