import os

from django.test import TestCase

from ..handle_files import download_file


class MockedResponse():
	def __init__(self, header_content_disposition: str, file_for_content: str) -> None:
		self.headers = {'content-disposition': header_content_disposition}

		with open(file_for_content, 'rb') as file:
			self.content = file.read()


class FileHandling(TestCase):
	def test_right_zip_file_is_provided_for_download(self):
		path_to_tar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'example_tar_archive.tar')
		mocked_response = MockedResponse('filename=archive_results_Mar14_07-32-14.tar', path_to_tar)
		response_file = download_file(mocked_response, True)

		assert 200 == response_file.status_code
		assert 'attachment; filename=archive_results_Mar14_07-32-14.zip' == response_file.headers['content-disposition']
		assert 'application/zip' == response_file.headers['content-type']

	def test_right_tar_file_is_provided_for_download(self):
		path_to_tar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'example_tar_archive.tar')
		mocked_response = MockedResponse('filename=archive_results_Mar14_07-32-14.tar', path_to_tar)
		response_file = download_file(mocked_response, False)

		assert 200 == response_file.status_code
		assert 'attachment; filename=archive_results_Mar14_07-32-14.tar' == response_file.headers['content-disposition']
		assert 'application/tar' == response_file.headers['content-type']
