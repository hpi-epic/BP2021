from unittest.mock import patch

from django.contrib.auth.models import User
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import TestCase
from django.test.client import RequestFactory

from ..api_response import APIResponse
from ..buttons import ButtonHandler
from ..models.config import Config
from ..models.container import Container, update_container
from .constant_tests import EXAMPLE_HIERARCHY_DICT, EXAMPLE_POST_REQUEST_ARGUMENTS


class ButtonTests(TestCase):
	def setUp(self):
		# get a container for testing
		config_object = Config.objects.create()
		self.user = User.objects.create(username='test_user', password='top_secret')
		self.test_container = Container.objects.create(
								command='training',
								id='1234',
								created_at='01.01.1970',
								last_check_at='now',
								name='test_container',
								config=config_object,
								user=self.user
								)

	def test_health_button(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request('/details', 'health')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'status': 'healthy :)'})

			test_button_handler.do_button_click()

			expected_arguments = self._get_expected_arguments('details.html', request)

			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			get_request_mock.assert_called_once()
			assert expected_arguments == actual_arguments
			assert 'healthy :)' == Container.objects.get(id='1234').health_status

	def test_pause_button(self):
		# mock a request that is send when user presses a button
		request = self._setup_request('/details', 'pause')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'status': 'paused'})

			test_button_handler.do_button_click()

			expected_arguments = self._get_expected_arguments('details.html', request)

			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments
			assert 'paused' == Container.objects.get(id='1234').health_status

	def test_unpause_button(self):
		# mock a request that is send when user presses a button
		request = self._setup_request('/details', 'pause')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'status': 'running'})

			test_button_handler.do_button_click()

			expected_arguments = self._get_expected_arguments('details.html', request)

			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments
			assert 'running' == Container.objects.get(id='1234').health_status

	def test_logs_button(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request('/details', 'logs')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'data': '1. test\n2. test'})

			test_button_handler.do_button_click()

			expected_arguments = self._get_expected_arguments('details.html', request, '2. test\n1. test')
			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments

	def test_tensorboard_button(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request('/details', 'data/tensorboard')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('details.html', request)

		with patch('alpha_business_app.buttons.redirect') as redirect_mock, \
			patch('alpha_business_app.buttons.send_get_request') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', '200', {'id': '1234', 'data': '6006'})

			test_button_handler.do_button_click()

			redirect_mock.assert_called_once_with('http://vm-midea03.eaalab.hpi.uni-potsdam.de:6006')

	def test_stop_button(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request('/observe', 'remove')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('observe.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.stop_container') as get_request_mock:
			content_dict = {'id': '1234', 'data': 'You successfully stopped the container'}
			get_request_mock.return_value = APIResponse('success', '200', content_dict)

			test_button_handler.do_button_click()

			expected_arguments = self._get_expected_arguments('observe.html', request, None, 'success', content_dict)
			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments

	def test_delete_button(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request('/observe', 'delete')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('observe.html', request)

		with patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.stop_container') as get_request_mock:
			content_dict = {'id': '1234', 'data': 'You successfully stopped the container'}
			get_request_mock.return_value = APIResponse('success', '200', content_dict)

			test_button_handler.do_button_click()

			expected_arguments = (request, 'observe.html',
						{'all_saved_containers': [],
							'container': None,
							'data': None,
							'success': 'You successfully removed all data'})
			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			render_mock.assert_called_once()
			assert expected_arguments == actual_arguments

	def test_download_data_from_archived(self):
		self.test_container.health_status = 'archived'
		update_container('1234', {'health_status': 'archived'})

		# mock a request that is sent when user presses a button
		request = self._setup_request('/download', 'data')

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('download.html', request)
		with patch('alpha_business_app.buttons.render') as render_mock:
			test_button_handler.do_button_click()

			expected_arguments = self._get_expected_arguments(view='download.html',
					request=request,
					data=None,
					keyword='error',
					keyword_data='You cannot download data from archived containers')

			render_mock.assert_called_once()
			actual_arguments = render_mock.call_args.args
			# cast the query set to list as well
			actual_arguments[2]['all_saved_containers'] = list(actual_arguments[2]['all_saved_containers'])

			assert expected_arguments == actual_arguments, f'\nExpected_arguments: {expected_arguments} \n actual aguments: {actual_arguments}'

	def test_download_zip_data(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request_with_parameters('/download', 'data', {'file_type': 'zip'})

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('download.html', request)
		with patch('alpha_business_app.buttons.download_file') as download_file_mock, \
			patch('alpha_business_app.buttons.send_get_request_with_streaming') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', content='test_content')
			test_button_handler.do_button_click()

			download_file_mock.assert_called_once_with('test_content', True, self.test_container)

	def test_download_tar_data(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request_with_parameters('/download', 'data', {'file_type': 'tar'})

		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('download.html', request)
		with patch('alpha_business_app.buttons.download_file') as download_file_mock, \
			patch('alpha_business_app.buttons.send_get_request_with_streaming') as get_request_mock:
			get_request_mock.return_value = APIResponse('success', content='test_content')
			test_button_handler.do_button_click()

			download_file_mock.assert_called_once_with('test_content', False, self.test_container)

	def test_start_button(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request_with_parameters('/configurator', 'start', EXAMPLE_POST_REQUEST_ARGUMENTS)
		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('configurator.html', request, rendering='config')
		with patch('alpha_business_app.buttons.send_post_request') as post_request_mock, \
			patch('alpha_business_app.buttons.redirect') as redirect_mock:
			api_response_dict = {
				'0': {
					'id': '2cc1fcd41e69f60055962e89c764f5c442cb2f4b76a9c4c8316c2bb9a5ffcdc6',
					'status': 'running',
					'data': 6006,
					'stream': None
				}, '1': {
					'id': 'ca166cff9b83bee9791b435e378574e52d71150c0f790df4fe793d02a86e031f',
					'status': 'sleeping',
					'data': 6007,
					'stream': None
				}
			}
			post_request_mock.return_value = APIResponse('success', content=api_response_dict)

			test_button_handler.do_button_click()
			post_request_mock.assert_called_once_with('start', EXAMPLE_HIERARCHY_DICT, {'num_experiments': 2})
			redirect_mock.assert_called_once_with('/observe', {'success': 'You successfully launched an experiment'})

		# assert config exists
		config_object = Config.objects.all()[1]
		assert 'Config for test_experiment' == config_object.name

		# assert two container were created
		container_set = Container.objects.filter(id='2cc1fcd41e69f60055962e89c764f5c442cb2f4b76a9c4c8316c2bb9a5ffcdc6')
		assert 1 == len(container_set)
		container1 = container_set[0]
		assert container1
		assert 'test_experiment (0)' == container1.name
		assert 'running' == container1.health_status

		container_set = Container.objects.filter(id='ca166cff9b83bee9791b435e378574e52d71150c0f790df4fe793d02a86e031f')
		assert 1 == len(container_set)
		container2 = container_set[0]
		assert container2
		assert 'test_experiment (1)' == container2.name
		assert 'sleeping' == container2.health_status

	def test_id_from_api_already_exists(self):
		# mock a request that is sent when user presses a button
		request = self._setup_request_with_parameters('/configurator', 'start', EXAMPLE_POST_REQUEST_ARGUMENTS)
		# setup a button handler for this request
		test_button_handler = self._setup_button_handler('configurator.html', request, rendering='config')
		with patch('alpha_business_app.buttons.send_post_request') as post_request_mock, \
			patch('alpha_business_app.buttons.render') as render_mock, \
			patch('alpha_business_app.buttons.stop_container'):
			api_response_dict = {
				'0': {
					'id': '1234',
					'status': 'running',
					'data': 6006,
					'stream': None
				}
			}
			post_request_mock.return_value = APIResponse('success', content=api_response_dict)

			test_button_handler.do_button_click()
			post_request_mock.assert_called_once_with('start', EXAMPLE_HIERARCHY_DICT, {'num_experiments': 2})
			render_mock.assert_called_once()
		assert 1 == len(Container.objects.all())

	def _setup_button_handler(self, view: str, request: RequestFactory, rendering: str = 'default') -> ButtonHandler:
		return ButtonHandler(request, view=view,
						container=self.test_container,
						rendering_method=rendering)

	def _setup_request(self, view: str, action: str) -> RequestFactory:
		request = RequestFactory().post(view, {'action': action, 'container_id': '1234'})
		request.user = self.user
		middleware = SessionMiddleware(request)
		middleware.process_request(request)
		request.session.save()
		return request

	def _setup_request_with_parameters(self, view: str, action: str, parameter: dict) -> RequestFactory:
		default_dict = {'action': action, 'container_id': '1234'}
		# if we switch to python 3.9+, we could also use default_dict | parameter here
		request = RequestFactory().post(view, {**default_dict, **parameter})
		request.user = self.user
		middleware = SessionMiddleware(request)
		middleware.process_request(request)
		request.session.save()
		return request

	def _get_expected_arguments(self, view: str, request: RequestFactory,
						data: str = None, keyword: str = None, keyword_data=None) -> tuple:
		# we need to cast it to list in order to compare, because query sets are not the same
		all_containers = list(Container.objects.all())

		# query the test container again, because it values might have changed
		self.test_container = Container.objects.get(id='1234')

		return (request,
				view,
				{
					'all_saved_containers': all_containers,
					'container': self.test_container,
					'data': data,
					keyword: keyword_data
				})
