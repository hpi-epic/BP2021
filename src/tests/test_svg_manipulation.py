import os
import re

import pytest

import configuration.utils_sim_market as ut
import monitoring.svg_manipulation as svg_manipulation


def teardown_function(function):
	print('***TEARDOWN***')
	for f in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring'):
		if re.match('test_svg_*', f):
			os.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + f)


def test_get_default_dict():
	default_dict = svg_manipulation.get_default_dict()
	for key, val in default_dict.items():
		if key != 'simulation_name' and key != 'simulation_episode_length':
			assert val == ''
	assert default_dict['simulation_name'] == 'Market Simulation'
	assert default_dict['simulation_episode_length'] == str(ut.EPISODE_LENGTH)


def test_replace_values():
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulation.replace_values('test_svg_replace_values')
	assert 'the passed filename must end in .svg: ' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulation.replace_values('test_svg_replace_values1.svg', {'simulation_name': 0})
	assert 'the dictionary should only contain strings: ' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		svg_manipulation.replace_values('test_svg_replace_values2.svg')
		svg_manipulation.replace_values('test_svg_replace_values2.svg')
	assert 'the specified file already exists: ' in str(assertion_message.value)
