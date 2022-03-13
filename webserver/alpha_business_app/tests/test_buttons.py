from unittest.mock import patch

from django.contrib.sessions.middleware import SessionMiddleware
from django.test import TestCase
from django.test.client import RequestFactory

from ..api_response import APIResponse
from ..buttons import ButtonHandler
from ..models import Container


def generate_mock_request():
	pass


class ButtonTests(TestCase):
	def setUp(self):
		# get a container for testing
		self.test_container = Container.objects.create(
								command='training',
								container_id='1234',
								created_at='01.01.1970',
								last_check_at='now',
								name='test_container'
								)

	def test_health(self):
		# mock a request that is send when user presses a button
		request = self.setup_request('/detail', 'health')

		# setup a button handler for this request
		test_button_handler = self.setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'status': 'healthy :)'})

			test_button_handler.do_button_click()

			expected_arguments = self.get_expected_arguments('details.html', request)

			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments

	def test_logs(self):
		# mock a request that is send when user presses a button
		request = self.setup_request('/detail', 'logs')

		# setup a button handler for this request
		test_button_handler = self.setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'data': '1. test\n2. test'})

			test_button_handler.do_button_click()

			expected_arguments = self.get_expected_arguments('details.html', request, '2. test\n1. test')
			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments

	def test_tensorboard(self):
		# mock a request that is send when user presses a button
		request = self.setup_request('/detail', 'data/tensorboard')

		# setup a button handler for this request
		test_button_handler = self.setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.redirect') as redirect_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'data': 'tensorboard_link:6006'})

			test_button_handler.do_button_click()

			redirect_mock.assert_called_once_with('tensorboard_link:6006')

	def test_stop(self):
		# mock a request that is send when user presses a button
		request = self.setup_request('/observe', 'remove')

		# setup a button handler for this request
		test_button_handler = self.setup_button_handler('observe.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.stop_container') as get_request_mock:
			content_dict = {'id': '1234', 'data': 'You successfully stopped the container'}
			get_request_mock.return_value = APIResponse('success', '200', content_dict)

			test_button_handler.do_button_click()

			expected_arguments = self.get_expected_arguments('observe.html', request, None, 'success', content_dict)
			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments

	# def test_remove(self):
	# 	# mock a request that is send when user presses a button
	# 	request = self.setup_request('/observe', 'delete')

	# 	# setup a button handler for this request
	# 	test_button_handler = self.setup_button_handler('observe.html', request)

	# 	with patch('alpha_business_app.buttons.render') as render_mock, \
	# 		patch('alpha_business_app.buttons.stop_container') as get_request_mock:
	# 		content_dict = {'id': '1234', 'data': 'You successfully stopped the container'}
	# 		get_request_mock.return_value = APIResponse('success', '200', content_dict)

	# 		test_button_handler.do_button_click()

	# 		expected_arguments = (request, 'observe.html',
	# 					{'all_saved_containers': [],
	# 						'container': self.test_container,
	# 						'data': None,
	# 						'success': 'You successfully removed all data'})
	# 		actual_arguments = render_mock.call_args.args
	# 		# cast the query set to list as well
	# 		actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

	# 		print(actual_arguments)
	# 		print(expected_arguments)
	# 		render_mock.assert_called_once()
	# 		assert expected_arguments == actual_arguments

	def setup_button_handler(self, view: str, request: RequestFactory) -> ButtonHandler:
		return ButtonHandler(request, view=view,
						container=self.test_container,
						rendering_method='default')

	def setup_request(self, view: str, action: str) -> RequestFactory:
		request = RequestFactory().post(view, {'action': action, 'container_id': '1234'})
		middleware = SessionMiddleware(request)
		middleware.process_request(request)
		request.session.save()
		return request

	def get_expected_arguments(self, view: str, request: RequestFactory,
						data: str = None, keyword: str = None, keyword_data=None) -> tuple:
		# we need to cast it to list in order to compare, because query sets are not the same
		all_containers = list(Container.objects.all())

		# query the test container again, because it values might have changed
		self.test_container = Container.objects.get(container_id='1234')

		return (request,
				view,
				{
					'all_saved_containers': all_containers,
					'container': self.test_container,
					'data': data,
					keyword: keyword_data
				})
