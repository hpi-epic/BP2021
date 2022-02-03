import mimetypes
import os
import tarfile

from django.http import HttpResponse

from .models import Container


def handle_uploaded_file(uploaded_config) -> None:
	path_to_configurations = './configurations'
	if not os.path.exists(path_to_configurations):
		os.mkdir(path_to_configurations)

	with open(os.path.join(path_to_configurations, uploaded_config.name), 'wb') as destination:
		for chunk in uploaded_config.chunks():
			destination.write(chunk)


def ensure_data_folder_structure(container_id: str) -> str:
	# make sure thet the folder ./data/<container_id> exists
	# in order to save all data belonging to this container in there
	data_folder = './data'
	path_to_container_data = os.path.join(data_folder, str(container_id))
	if os.path.exists(path_to_container_data):
		return path_to_container_data
	os.makedirs(path_to_container_data)
	# put the used config in the data folder
	config = Container.objects.get(container_id=container_id).config_file
	with open(os.path.join(path_to_container_data, 'config.json'), 'w') as config_file:
		config_file.write(config)

	return path_to_container_data


def save_data(response, container_id: str) -> str:
	container_data_folder = ensure_data_folder_structure(container_id)

	# save the archive with the filename from response in data folder of container
	archive_name = response.headers['content-disposition'][9:]
	path_to_archive = os.path.join(container_data_folder, archive_name)

	with open(path_to_archive, 'wb') as new_archive:
		new_archive.write(response.content)
	return path_to_archive


def add_config_to_tar_archive(path_to_file: str, container_id: str) -> None:
	# add config to tar file
	path_to_container_data = os.path.dirname(os.path.abspath(path_to_file))
	tf = tarfile.open(path_to_file, 'a')
	print(path_to_container_data)
	tf.add(os.path.join(path_to_container_data, 'config.json'))
	tf.close()


def download_file(path_to_file: str, container_id: str) -> HttpResponse:
	# get the mime type of the file
	mime_type, _ = mimetypes.guess_type(path_to_file)

	# if it is a tar, we can add the config file to the archive
	if mime_type == 'application/x-tar':
		add_config_to_tar_archive(path_to_file, container_id)

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
