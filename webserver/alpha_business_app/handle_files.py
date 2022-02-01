import mimetypes
import os
import time

from django.http import HttpResponse


def handle_uploaded_file(uploaded_config) -> None:
	path_to_configurations = './configurations'
	if not os.path.exists(path_to_configurations):
		os.mkdir(path_to_configurations)

	with open(os.path.join(path_to_configurations, uploaded_config.name), 'wb') as destination:
		for chunk in uploaded_config.chunks():
			destination.write(chunk)


def save_data(response, container_id: str) -> str:
	# make sure thet the folder ./data/<container_id> exists
	# in order to save all data belonging to this container in there
	data_folder = './data'
	if not os.path.exists(data_folder):
		os.mkdir(data_folder)
	container_data_folder = os.path.join(data_folder, str(container_id))
	if not os.path.exists(container_data_folder):
		os.mkdir(container_data_folder)

	archive_name = time.strftime('%b%d_%H-%M-%S') + '.tar'
	path_to_archive = os.path.join(container_data_folder, archive_name)
	with open(path_to_archive, 'wb') as new_archive:
		new_archive.write(response.content)
	return path_to_archive


def download_file(path_to_file) -> HttpResponse:
	# get the mime type of the file
	mime_type, _ = mimetypes.guess_type(path_to_file)
	# open file and write to HttpResponse
	with open(path_to_file, 'rb') as archive:
		response = HttpResponse(archive, content_type=mime_type)
	# set the HTTP header for sending to browser
	archive_name = os.path.basename(path_to_file)
	container_id = os.path.basename(os.path.dirname(path_to_file))
	file_name = 'container_' + container_id + '_' + archive_name
	response['Content-Disposition'] = 'attachment; filename=%s' % file_name
	# Return the response value
	return response
