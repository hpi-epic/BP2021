import mimetypes
import os
import shutil
import tarfile
import time

from django.http import HttpResponse

from .constants import CONFIGURATION_DIR, DATA_DIR
from .models import Container


def archive_files(container_id: str) -> HttpResponse:
	"""
	This will get you ona archive of all the archives in the data folder of the container.

	Args:
		container_id (str): The id of the wanted container.

	Returns:
		HttpResponse: All files beloning to this container.
	"""
	container_data_path = os.path.join(DATA_DIR, container_id)
	archive_path = os.path.join(container_data_path, f'all_{time.strftime("%b%d_%H-%M-%S")}.tar')

	tar_archive = tarfile.open(archive_path, 'x')
	tar_archive.close()
	files_to_be_included = [file for file in os.listdir(container_data_path) if not file.startswith('all')]

	_add_files_to_archive(archive_path, container_data_path, files_to_be_included)
	return _file_as_http_response(archive_path)


def download_file(path_to_file: str) -> HttpResponse:
	"""
	Makes a file available to the user, if the file is a tar archive it adds the `config.json` to it.

	Args:
		path_to_file (str): Path to the file that should be downloaded.

	Returns:
		HttpResponse: A response including the file with all headers set for the user to save it.
	"""
	# get the mime type of the file
	mime_type, _ = mimetypes.guess_type(path_to_file)

	# if it is a tar, we can add the config file to the archive
	if mime_type == 'application/x-tar':
		path_to_container_data = os.path.dirname(path_to_file)
		_add_files_to_archive(path_to_file, path_to_container_data, ['config.json'])
	print('download file:', path_to_file)
	return _file_as_http_response(path_to_file)


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


def save_data(response, container_id: str) -> str:
	"""
	Saves a tar file to the data folder of the container.

	Args:
		response (APIResponse): A converted response from the API.
		container_id (str): The container id the data belongs to.

	Returns:
		str: path to the saved folder.
	"""
	container_data_folder = _ensure_data_folder_structure(container_id)

	# save the archive with the filename from response in data folder of container
	archive_name = response.headers['content-disposition'][9:]
	path_to_archive = os.path.join(container_data_folder, archive_name)
	print(archive_name)

	with open(path_to_archive, 'wb') as new_archive:
		new_archive.write(response.content)
	print('save:', path_to_archive)
	return path_to_archive


def _add_files_to_archive(path_to_tar_archive: str, path_to_files: str, files: list) -> None:
	"""
	Adds all given files to the given archive

	Args:
		path_to_tar_archive (str): Path to the archive the files should be added.
		path_to_files (str): path to the files that should be added, must match the list of files.
		files (list): all files that should be added to the archive.
	"""
	tar_archive = tarfile.open(path_to_tar_archive, 'a')
	for file in files:
		tar_archive.add(os.path.join(path_to_files, file), arcname=file)
	tar_archive.close()


def _convert_tar_file_to_zip(path_to_tar: str) -> str:
	path_to_temp_folder = path_to_tar[:-4]

	# extract all files to a temp folder
	# extractall from the tarfile libary throws an error, so we use a commandline tool here
	os.mkdir(path_to_temp_folder)
	os.system(f'tar -xvf {path_to_tar} -C {path_to_temp_folder}')

	shutil.make_archive(path_to_temp_folder, 'zip', path_to_temp_folder)
	try:
		shutil.rmtree(path_to_temp_folder)
	except Exception:
		print(f'deleting {path_to_temp_folder} didn\'t work, please check')

	return path_to_temp_folder + '.zip'


def _ensure_data_folder_structure(container_id: str) -> str:
	"""
	Makes sure thet the folder ./data/<container_id> exists, in order to save all data belonging to this container in there.

	Args:
		container_id (str): id of the container

	Returns:
		str: path to the data folder of the requested container.
	"""
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


def _file_as_http_response(path_to_archive: str) -> HttpResponse:
	"""
	Converts a given file into an HttpResponse and guesses mime type of file.

	Args:
		path_to_file (str): path to the file that should be converted.

	Returns:
		HttpResponse: HttpResponse containing the file.
	"""
	# currently we want to download zip files, so convert to zip file
	print('in file_as_http:', path_to_archive)
	path_to_file = _convert_tar_file_to_zip(path_to_archive)

	mime_type, _ = mimetypes.guess_type(path_to_file)

	# open file and write to HttpResponse
	with open(path_to_file, 'rb') as archive:
		response = HttpResponse(archive, content_type=mime_type)

	# set the HTTP header for sending to browser
	archive_name = os.path.basename(path_to_file)
	container_id = os.path.basename(os.path.dirname(path_to_file))
	container_name = Container.objects.get(container_id=container_id).name
	file_name = f'container_{container_name}_{archive_name}'
	response['Content-Disposition'] = f'attachment; filename={file_name}'
	# Return the response value
	return response
