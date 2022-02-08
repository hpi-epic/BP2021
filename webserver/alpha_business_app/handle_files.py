import mimetypes
import os
import tarfile
import time

from django.http import HttpResponse

from .constants import CONFIGURATION_DIR, DATA_DIR
from .models import Container


def archive_files(container_id: str) -> str:
	container_data_path = os.path.join(DATA_DIR, container_id)
	archive_path = os.path.join(container_data_path, 'all_' + time.strftime('%b%d_%H-%M-%S') + '.tar')

	tar_archive = tarfile.open(archive_path, 'x')
	tar_archive.close()
	files_to_be_included = [file for file in os.listdir(container_data_path) if not file.startswith('all')]

	_add_files_to_archive(archive_path, container_data_path, files_to_be_included)
	return _file_as_http_response(archive_path, 'application/x-tar')


def handle_uploaded_file(uploaded_config) -> None:
	path_to_configurations = CONFIGURATION_DIR
	if not os.path.exists(path_to_configurations):
		os.mkdir(path_to_configurations)

	with open(os.path.join(path_to_configurations, uploaded_config.name), 'wb') as destination:
		for chunk in uploaded_config.chunks():
			destination.write(chunk)


def save_data(response, container_id: str) -> str:
	container_data_folder = _ensure_data_folder_structure(container_id)

	# save the archive with the filename from response in data folder of container
	archive_name = response.headers['content-disposition'][9:]
	path_to_archive = os.path.join(container_data_folder, archive_name)

	with open(path_to_archive, 'wb') as new_archive:
		new_archive.write(response.content)
	return path_to_archive


def download_file(path_to_file: str) -> HttpResponse:
	# get the mime type of the file
	mime_type, _ = mimetypes.guess_type(path_to_file)

	# if it is a tar, we can add the config file to the archive
	if mime_type == 'application/x-tar':
		path_to_container_data = os.path.dirname(path_to_file)
		_add_files_to_archive(path_to_file, path_to_container_data, ['config.json'])
	return _file_as_http_response(path_to_file, mime_type)


def _add_files_to_archive(path_to_tar_archive: str, path_to_files: str, files: list):
	tar_archive = tarfile.open(path_to_tar_archive, 'a')
	for file in files:
		tar_archive.add(os.path.join(path_to_files, file), arcname=file)
	tar_archive.close()


def _ensure_data_folder_structure(container_id: str) -> str:
	# make sure thet the folder ./data/<container_id> exists
	# in order to save all data belonging to this container in there
	data_folder = DATA_DIR
	path_to_container_data = os.path.join(data_folder, str(container_id))
	if os.path.exists(path_to_container_data):
		return path_to_container_data
	os.makedirs(path_to_container_data)
	# put the used config in the data folder
	config = Container.objects.get(container_id=container_id).config_file
	with open(os.path.join(path_to_container_data, 'config.json'), 'w') as config_file:
		config_file.write(config)

	return path_to_container_data


def _file_as_http_response(path_to_file: str, mime_type: str):
	# open file and write to HttpResponse
	with open(path_to_file, 'rb') as archive:
		response = HttpResponse(archive, content_type=mime_type)

	# set the HTTP header for sending to browser
	archive_name = os.path.basename(path_to_file)
	container_id = os.path.basename(os.path.dirname(path_to_file))
	container_name = Container.objects.get(container_id=container_id).name
	file_name = 'container_' + container_name + '_' + archive_name
	response['Content-Disposition'] = 'attachment; filename=%s' % file_name
	# Return the response value
	return response
