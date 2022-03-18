import json
import os
import tarfile
import zipfile
from io import BytesIO

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from .models.config import *
from .models.config import EnvironmentConfig, HyperparameterConfig


def handle_uploaded_file(request, uploaded_config) -> None:

	if uploaded_config.name[-5:] != '.json':
		return render(request, 'upload.html', {'error': 'You can only upload files in JSON format.'})

	file_content = b''
	for chunk in uploaded_config.chunks():
		file_content += chunk

	try:
		content_as_dict = json.loads(file_content)
	except:
		return render(request, 'upload.html', {'error': 'Your JSON is not valid'})

	# hyperparameter_configs = []
	# environment_configs = []
	# for key in content_as_dict.keys():
	# 	if key in HYPERPARAMETER_CONFIG_KEYS:
	# 		hyperparameter_configs += [content_as_dict[key]]
	# 	elif key in ENVIRONMENT_CONFIG_KEYS:
	# 		environment_configs += [content_as_dict[key]]
	# 	else:
	# 		return render(request, 'upload.html', {'error': f'The key {key} is unknown'})

	parse_dict_to_database('hyperparameter', content_as_dict)

	return HttpResponseRedirect('/start_container', {'success': 'You successfully uploaded a config file'})


def parse_dict_to_database(name: str, content: dict):
	containing_dict = [(name, value) for name, value in content.items() if type(value) == dict]
	print('++++++++++++++++++++++++++++++++')
	print(name)
	print(containing_dict)
	# print(content)
	obj = []
	for keyword, config in containing_dict:
		obj += [(keyword, parse_dict_to_database(keyword, config))]

	not_containing_dict = dict([(name, value) for name, value in content.items() if type(value) != dict])
	# print(dict(not_containing_dict))
	config_class = _to_class_name(name)
	print(not_containing_dict)
	print(obj)
	for keyword, model_instance in obj:
		not_containing_dict[keyword] = model_instance

	print(globals()[config_class])
	print('####################')
	return globals()[config_class].objects.create(**not_containing_dict)

def _to_class_name(name: str) -> str:
	return ''.join([x.capitalize() for x in name.split('_')]) + 'Config'

def download_file(response, wants_zip: bool) -> HttpResponse:
	"""
	Makes the dat from the API available for the user and adds the config file before.
	This can either be a zip or a tarfile.

	Args:
		response (Response): Response from the API which is a tar archive.
		wants_zip (bool): Indicates whether the user wants to download the data as a zipped file.

	Returns:
		HttpResponse: response for the user containing the file.
	"""
	archive_name = response.headers['content-disposition'][9:-4]

	# convert tar file to file like object to be able to work with it in memory
	file_like_tar_archive = BytesIO(response.content)

	if wants_zip:
		zip_file = _convert_tar_file_to_zip(file_like_tar_archive)
		zip_file = _add_files_to_zip(zip_file, CONFIGURATION_DIR, ['hyperparameter_config.json'])
		fake_file = zip_file
		application_type = 'zip'
	else:
		tar_file = _add_files_to_tar(file_like_tar_archive, CONFIGURATION_DIR, ['hyperparameter_config.json'])
		fake_file = tar_file
		application_type = 'tar'

	# put together an http response for the browser
	file_response = HttpResponse(fake_file.getvalue(), content_type=f'application/{application_type}')
	file_response['Content-Disposition'] = f'attachment; filename={archive_name}.{application_type}'

	return file_response


def _add_files_to_tar(fake_tar_archive: BytesIO, path_to_add_files: str, files: list) -> BytesIO:
	"""
	This function adds the given files at the path to the given tar archive.

	Args:
		fake_tar_archive (BytesIO): the archive the files should be added.
		path_to_add_files (str): path to the files that need to be added.
		files (list): names of the files at the path that need to be added.
	"""
	# TODO: assert all path + file exist
	print(f'adding {files} to tar archive')
	tar_archive = tarfile.open(fileobj=fake_tar_archive, mode='a:')
	for file in files:
		tar_archive.add(os.path.join(path_to_add_files, file), arcname=file)
	tar_archive.close()

	return fake_tar_archive


def _add_files_to_zip(file_like_zip: BytesIO, path_to_add_files: str, files: list) -> BytesIO:
	"""
	This function adds the given files at the path to the given tar archive.

	Args:
		file_like_zip (BytesIO): the zip archive the files should be added.
		path_to_add_files (str): path to the files that need to be added.
		files (list): names of the files at the path that need to be added.
	"""
	# TODO: assert all path + file exist
	print(f'adding {files} to zip archive')
	zip_archive = zipfile.ZipFile(file=file_like_zip, mode='a', compression=zipfile.ZIP_DEFLATED)
	for file in files:
		zip_archive.write(os.path.join(path_to_add_files, file), arcname=file)
	zip_archive.close()

	return file_like_zip


def _convert_tar_file_to_zip(fake_tar_archive: BytesIO) -> BytesIO:
	"""
	Converts a tar file into a zip file.

	Args:
		fake_tar_archive (BytesIO): bytes of a tar archive

	Returns:
		BytesIO: fake file bytes of zip archive.
	"""
	print('converting tar archive to zip')
	tar_archive = tarfile.open(fileobj=fake_tar_archive, mode='r:')

	# create an in memory zip file
	file_like_zip = BytesIO()
	zip_archive = zipfile.ZipFile(file=file_like_zip, mode='a', compression=zipfile.ZIP_DEFLATED)

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
