import json
import os
import tarfile
import zipfile
from io import BytesIO

from django.http import HttpResponse
from django.shortcuts import redirect, render

from .constants import CONFIGURATION_DIR
from .models.config import *


# https://stackoverflow.com/questions/14902299/json-loads-allows-duplicate-keys-in-a-dictionary-overwriting-the-first-value
def dict_raise_on_duplicates(ordered_pairs):
	"""Reject duplicate keys."""
	d = {}
	for k, v in ordered_pairs:
		if k in d:
			raise ValueError('Your config contains duplicate keys: %r' % (k,))
		else:
			d[k] = v
	return d


def handle_uploaded_file(request, uploaded_config) -> None:
	# we only accept json files
	if uploaded_config.name[-5:] != '.json':
		return render(request, 'upload.html', {'error': 'You can only upload files in JSON format.'})

	# read the file content
	file_content = b''
	for chunk in uploaded_config.chunks():
		file_content += chunk

	# try to convert the file content to dict
	try:
		content_as_dict = json.loads(file_content, object_pairs_hook=dict_raise_on_duplicates)
	except json.JSONDecodeError:
		return render(request, 'upload.html', {'error': 'Your JSON is not valid'})
	except ValueError as value:
		return render(request, 'upload.html', {'error': str(value)})

	# figure out which parts of the config file belong to hyperparameter or environment config
	hyperparameter_fields = get_config_field_names(HyperparameterConfig)
	environment_fields = get_config_field_names(EnvironmentConfig)

	# figure out which keywords belong to hyperparameter and which keywords belong to environment
	hyperparameter_configs = {}
	environment_configs = {}
	contains_hyperparameter = False
	contains_environment = False
	for key in content_as_dict.keys():
		if key in hyperparameter_fields:
			hyperparameter_configs[key] = content_as_dict[key]
			contains_hyperparameter = True
		elif key in environment_fields:
			environment_configs[key] = content_as_dict[key]
			contains_environment = True
		else:
			return render(request, 'upload.html', {'error': f'The key {key} is unknown'})
	# TODO: outsource the checking to the pip package?
	# check if datatypes are correct.
	# check if all keys in the dictionaries are valid
	status, error_msg = check_dict_keys('environment', environment_configs)
	if not status:
		return render(request, 'upload.html', {'error': error_msg})
	status, error_msg = check_dict_keys('hyperparameter', hyperparameter_configs)
	if not status:
		return render(request, 'upload.html', {'error': error_msg})

	hyperparameter_config = None
	environment_config = None
	if contains_hyperparameter is True:
		hyperparameter_config = parse_dict_to_database('hyperparameter', hyperparameter_configs)
	if contains_environment is True:
		environment_config = parse_dict_to_database('environment', environment_configs)

	Config.objects.create(environment=environment_config, hyperparameter=hyperparameter_config)
	return redirect('/start_container', {'success': 'You successfully uploaded a config file'})


def check_dict_keys(name: str, content: dict) -> tuple:
	if name == 'agents':
		status, error_msg = check_agents(content)
		if not status:
			return False, error_msg
		else:
			return True, ''
	containing_dict = [(name, value) for name, value in content.items() if type(value) == dict]

	for keyword, config in containing_dict:
		status, error_msg = check_dict_keys(keyword, config)
		if not status:
			return False, error_msg

	class_name = to_config_class_name(name)
	allowed_keys = get_config_field_names(globals()[class_name])
	used_keys = list(content.keys())
	difference = list(set(used_keys) - set(allowed_keys))
	if difference != []:
		return False, f'The keyword(s) {difference} are not allowed in {name}'

	return True, ''


def check_agents(content: dict) -> tuple:
	for agent_name, agent_parameters in content.items():
		agent_class = to_config_class_name(agent_name)
		allowed_keys = get_config_field_names(globals()[agent_class])
		used_keys = list(agent_parameters.keys())
		difference = list(set(used_keys) - set(allowed_keys))
		if difference != []:
			return False, f'The keyword(s) {difference} are not allowed in {agent_name}'
	return True, ''


def parse_dict_to_database(name: str, content: dict):
	if name == 'agents':
		# since django does only support many-to-one relationships (not one-to-many),
		# we need to parse the agents slightly different, to be able to reference many agents with the agents keyword
		return parse_agents(content)
	# get all key value pairs, that contain another dict
	containing_dict = [(name, value) for name, value in content.items() if type(value) == dict]
	# loop through of these pairs, in order to parse these dictionaries and add
	# the parsed sub-element to the current element
	sub_elements = []
	for keyword, config in containing_dict:
		sub_elements += [(keyword, parse_dict_to_database(keyword, config))]

	# get all elements that do not contain another dictionary
	not_containing_dict = dict([(name, value) for name, value in content.items() if type(value) != dict])

	# add the sub-elements to the dictionary with the other key value pairs not containing another dictionary
	for keyword, model_instance in sub_elements:
		not_containing_dict[keyword] = model_instance

	# figure out which config object to create and return the created objects
	config_class = to_config_class_name(name)
	return _create_object_from(config_class, not_containing_dict)


def parse_agents(agents_in_dict: dict) -> AgentsConfig:
	agents = AgentsConfig.objects.create()
	for agent_name, agent_parameters in agents_in_dict.items():
		agent_parameters['agents_config'] = agents
		_create_object_from(to_config_class_name(agent_name), agent_parameters)
	return agents


def _create_object_from(class_name: str, parameters: dict):
	return globals()[class_name].objects.create(**parameters)


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
