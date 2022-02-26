import os
import tarfile
import zipfile
from io import BytesIO

from django.http import HttpResponse

from .constants import CONFIGURATION_DIR


def handle_uploaded_file(uploaded_config) -> None:
	"""
	Writes an uploaded config to our internal storage.

	Args:
		uploaded_config (InMemoryUploadedFile): The file the user just uploaded.
	"""
	path_to_configurations = CONFIGURATION_DIR
	if not os.path.exists(path_to_configurations):
		os.mkdir(path_to_configurations)

	with open(os.path.join(path_to_configurations, uploaded_config.name), 'wb') as destination:
		for chunk in uploaded_config.chunks():
			destination.write(chunk)


def download_file(response) -> HttpResponse:
	archive_name = response.headers['content-disposition'][9:-4]

	# convert tar file to file like object to be able to convert it to zip in memory
	file_like_tar_archive = BytesIO(response.content)
	# file_like_tar_archive = _add_files_to_archive(file_like_tar_archive, CONFIGURATION_DIR, ['config.json'])

	zip_file = _convert_tar_file_to_zip(file_like_tar_archive)

	# put together an http response for the browser
	response = HttpResponse(zip_file.getvalue(), content_type='application/zip')
	response['Content-Disposition'] = f'attachment; filename={archive_name}.zip'

	return response


def _add_files_to_archive(fake_tar_archive: BytesIO, path_to_add_files: str, files: list) -> BytesIO:
	"""
	This function adds the given files at the path to the given tar archive.

	Args:
		fake_tar_archive (BytesIO): the archive the files should be added.
		path_to_add_files (str): path to the files that need to be added.
		files (list): names of the files at the path that need to be added.
	"""
	# TODO: assert all path + file exist
	print(f'adding {files} to tar archive')
	tar_archive = tarfile.open(fileobj=fake_tar_archive, mode='a')
	for file in files:
		tar_archive.add(os.path.join(path_to_add_files, file), arcname=file)
	tar_archive.close()
	return fake_tar_archive


def _convert_tar_file_to_zip(fake_tar_archive: BytesIO) -> BytesIO:
	print('converting tar archive to zip')
	tar_archive = tarfile.open(fileobj=fake_tar_archive, mode='r')

	# create an in memory zip file
	file_like_zip = BytesIO()
	zip_archive = zipfile.ZipFile(file=file_like_zip, mode='a', compression=zipfile.ZIP_DEFLATED)

	# iterate over all files in the tarfile
	for member in tar_archive:
		file = tar_archive.extractfile(member)
		if file:
			# file will be None for empty directories
			file_content = file.read()
			file_name = member.name
			zip_archive.writestr(file_name, file_content)
	zip_archive.close()
	tar_archive.close()

	return file_like_zip
