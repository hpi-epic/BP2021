from django.test import TestCase

from ..api_response import APIResponse
from ..handle_requests import _error_handling_API


class MockResponse():
	def __init__(self, status_code: int = 200) -> None:
		self.status_code = status_code

	def json(self):
		return {
			'id': 'testcontainer1234567890abcdef',
			'status': 'exited (0)',
			'data': None,
			'stream': None
			}


class ContainerTest(TestCase):
	def test_error_handling(self):
		mocked_api_response404 = MockResponse(404)
		expected_response404 = APIResponse('error',
					content=mocked_api_response404.json()['status'],
					http_status=mocked_api_response404.status_code)

		# mocked_api_response500 = MockResponse(500)

		converted_response = _error_handling_API(mocked_api_response404)

		assert expected_response404.http_status == converted_response.http_status
		assert expected_response404.status_code == converted_response.status_code
		assert expected_response404.content == converted_response.content
