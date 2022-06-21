from django.test import TestCase

from ..adjustable_fields import get_rl_parameter_prefill


class AdjustableFieldsTests(TestCase):
	def test_rl_hyperparameter_with_prefill(self):
		prefill_dict = {'gamma': 0.9, 'learning_rate': 0.4, 'test': None}
		error_dict = {'gamma': 'test', 'learning_rate': None, 'test': None}
		expected_list = [
			{'name': 'gamma', 'prefill': 0.9, 'error': 'test'},
			{'name': 'learning_rate', 'prefill': 0.4, 'error': ''},
			{'name': 'test', 'prefill': '', 'error': ''}
		]
		actual_list = get_rl_parameter_prefill(prefill_dict, error_dict)
		assert actual_list == expected_list
