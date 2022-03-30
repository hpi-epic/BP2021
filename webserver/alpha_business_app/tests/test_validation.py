from django.test import TestCase

from ..validation import check_agents


class FileHandling(TestCase):
	def test_valid_agents(self):
		status, error_msg = check_agents({'Rule_Based Agent':
		{'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'},
			'CE Rebuy Agent (QLearning)': {'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
				'argument': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'}})

		assert status is True
		assert '' == error_msg

	def test_invalid_agents(self):
		status, error_msg = check_agents({'Rule_Based Agent': {'test': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'}})

		assert status is False
		assert "The keyword(s) ['test'] are not allowed in Rule_Based Agent" == error_msg

	# def test_invalid_values(self):
	# 	test_uploaded_file = MockedUploadedFile('config.json', b'{"rl": {"gamma": "bla"}')
	# 	with patch('alpha_business_app.handle_files.render') as render_mock:
	# 		handle_uploaded_file('this is not important', test_uploaded_file)

	# 		actual_arguments = render_mock.call_args.args

	# 		render_mock.assert_called_once()
	# 		assert 'upload.html' == actual_arguments[1]
	# 		assert {'error': ''} == actual_arguments[2]
