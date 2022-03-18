import os
from unittest.mock import patch

from django.contrib.sessions.middleware import SessionMiddleware
from django.test import TestCase
from django.test.client import RequestFactory

from ..handle_files import download_file, handle_uploaded_file


class MockedResponse():
	def __init__(self, header_content_disposition: str, file_for_content: str) -> None:
		self.headers = {'content-disposition': header_content_disposition}

		with open(file_for_content, 'rb') as file:
			self.content = file.read()


class MockedUploadedFile():
	def __init__(self, _name: str, _content: str) -> None:
		self.name = _name
		self.content = _content

	def chunks(self):
		return [self.content]

class FileHandling(TestCase):
	def test_right_zip_file_is_provided_for_download(self):
		path_to_tar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'example_tar_archive.tar')
		mocked_response = MockedResponse('filename=archive_results_Mar14_07-32-14.tar', path_to_tar)
		with patch('alpha_business_app.handle_files._add_files_to_zip'):
			response_file = download_file(mocked_response, True)

		assert 200 == response_file.status_code
		assert 'attachment; filename=archive_results_Mar14_07-32-14.zip' == response_file.headers['content-disposition']
		assert 'application/zip' == response_file.headers['content-type']

	def test_right_tar_file_is_provided_for_download(self):
		path_to_tar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'example_tar_archive.tar')
		mocked_response = MockedResponse('filename=archive_results_Mar14_07-32-14.tar', path_to_tar)
		with patch('alpha_business_app.handle_files._add_files_to_tar'):
			response_file = download_file(mocked_response, False)

		assert 200 == response_file.status_code
		assert 'attachment; filename=archive_results_Mar14_07-32-14.tar' == response_file.headers['content-disposition']
		assert 'application/tar' == response_file.headers['content-type']

	def test_uploaded_file_is_not_json(self):
		test_uploaded_file = MockedUploadedFile('test_file.jpg', b'this is a jpg')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('request', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'You can only upload files in JSON format.'} == actual_arguments[2]

	def test_uploaded_file_invalid_json(self):
		test_uploaded_file = MockedUploadedFile('test_file.json', b'{ "rl": "1234"')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('request', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'Your JSON is not valid'} == actual_arguments[2]

	def test_uploaded_file_with_unknown_key(self):
		test_uploaded_file = MockedUploadedFile('test_file.json', b'{ "test": "1234" }')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('request', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'The key test is unknown'} == actual_arguments[2]
