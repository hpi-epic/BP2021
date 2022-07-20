import csv
import json
import re
import tarfile
import zipfile
from io import BytesIO

from django.http import HttpResponse
from django.shortcuts import redirect, render

from recommerce.configuration.config_validation import validate_config

from .config_parser import ConfigModelParser
from .models.config import Config
from .models.container import Container


# https://stackoverflow.com/questions/14902299/json-loads-allows-duplicate-keys-in-a-dictionary-overwriting-the-first-value
def _dict_raise_on_duplicates(ordered_pairs):
	"""
	Reject duplicate keys.
	"""
	no_duplicates_dict = {}
	for key, value in ordered_pairs:
		if key in no_duplicates_dict:
			raise ValueError(f"Your config contains duplicate keys: '{key}'")
		else:
			no_duplicates_dict[key] = value
	return no_duplicates_dict


def handle_uploaded_file(request, uploaded_config) -> HttpResponse:
	"""
	Checks if an uploaded config file is valid and parses it to the datastructure.

	Args:
		request (Request): post request by the user
		uploaded_config (InMemoryUploadedFile): by user uploaded config file
		filename (str, optional): the filename of the uploaded file. Defaults to ''.

	Returns:
		HttpResponse: either a redirect to the configurator or a render for the upload with an error message
	"""
	# we only accept json files
	if uploaded_config.name[-5:] != '.json':
		return render(request, 'upload.html', {'error': 'You can only upload files in JSON format.'})

	# read the file content
	file_content = b''
	for chunk in uploaded_config.chunks():
		file_content += chunk

	# try to convert the file content to dict
	try:
		content_as_dict = json.loads(file_content, object_pairs_hook=_dict_raise_on_duplicates)
	except json.JSONDecodeError:
		return render(request, 'upload.html', {'error': 'Your JSON is not valid'})
	except ValueError as value:
		return render(request, 'upload.html', {'error': str(value)})

	# Validate the config file using the recommerce validation functionality
	validate_status, validate_data = validate_config(content_as_dict)
	if not validate_status:
		return render(request, 'upload.html', {'error': validate_data})

	# configs and their corresponding top level keys as list
	config_objects = _get_top_level_and_configs(validate_data)

	# parse config model to datastructure
	parser = ConfigModelParser()
	resulting_config_parts = []
	for top_level, config in config_objects:
		try:
			resulting_config_parts += [(top_level, parser.parse_config_dict_to_datastructure(top_level, config))]
		except ValueError:
			return render(request, 'upload.html', {'error': 'Your config is wrong'})
		except TypeError as error:
			invalid_keyword_search = re.search('.*keyword argument (.*)', str(error))
			return render(request, 'upload.html', {'error': f'Your config contains an invalid key: {invalid_keyword_search.group(1)}'})

	# Make it a real config object
	environment_config, hyperparameter_config = _get_config_parts(resulting_config_parts)
	given_name = request.POST['config_name']
	config_name = given_name if given_name else uploaded_config.name
	Config.objects.create(environment=environment_config, hyperparameter=hyperparameter_config, name=config_name, user=request.user)
	return redirect('/configurator', {'success': 'You successfully uploaded a config file'})


def download_config(wanted_container: Container) -> HttpResponse:
	"""
	Provides the configuration file of the given container as an HttpResponse.

	Args:
		wanted_container (Container): container, which the configuration belongs to

	Returns:
		HttpResponse: contains the configuration file belonging to this container.
	"""
	config_object = wanted_container.config
	config_as_string = json.dumps(config_object.as_dict(), indent=4, sort_keys=True)
	file_response = HttpResponse(config_as_string, content_type='application/json')
	# we only want the alphanumeric characters from container name
	file_name = ''.join(character for character in wanted_container.name if character.isalnum())
	file_response['Content-Disposition'] = f'attachment; filename=config_{file_name}.json'
	return file_response


def download_file(response, wants_zip: bool, wanted_container: Container) -> HttpResponse:
	"""
	Makes the data from the API available for the user and adds the config file before.
	This can either be a zip or a tarfile.

	Args:
		response (Response): Response from the API which is a tar archive.
		wants_zip (bool): Indicates whether the user wants to download the data as a zipped file.
		wanted_container (Container): The container the data belongs to is necessary because we need the config file from it

	Returns:
		HttpResponse: response for the user containing the file.
	"""
	archive_name = response.headers['content-disposition'][9:-4]

	# get the config file from the container in order to add it to the archive
	config_object = wanted_container.config
	config_as_string = json.dumps(config_object.as_dict(), indent=4, sort_keys=True)

	# convert tar file to file like object to be able to work with it in memory
	file_like_tar_archive = BytesIO(response.content)

	if wants_zip:
		zip_file = _convert_tar_file_to_zip(file_like_tar_archive)
		zip_file = _add_files_to_zip(zip_file, config_as_string)
		fake_file = zip_file
		application_type = 'zip'
	else:
		fake_file = file_like_tar_archive
		application_type = 'tar'

	# put together an http response for the browser
	file_response = HttpResponse(fake_file.getvalue(), content_type=f'application/{application_type}')
	file_response['Content-Disposition'] = f'attachment; filename={archive_name}.{application_type}'

	return file_response


def get_statistic_data(csv_string: str, file_name: str) -> HttpResponse:
	"""
	Converts string in csv format (seperator = ;) to actual csv file and returns this file as HttpResponse.
	Warning, we do not consider quoting! The string will be split at \n and ;
	Args:
		csv_string (str): string in csv format (; seperated)
		file_name (str): name of final csv file

	Returns:
		HttpResponse: contains csv file
	"""
	lines = csv_string.split('\n')
	final_lines = []
	for line in lines:
		final_lines += [line.split(';')]
	response = HttpResponse(
		content_type='text/csv',
		headers={'Content-Disposition': f'attachment; filename="{file_name}.csv"'},
	)
	writer = csv.writer(response, delimiter=';', quotechar='"')
	writer.writerows(final_lines)
	return response


def _add_files_to_zip(file_like_zip: BytesIO, string_to_add: str) -> BytesIO:
	"""
	Adds a string as `config.json` to the given zip archive.

	Args:
		file_like_zip (BytesIO): in memory file
		string_to_add (str): the string which should be added in `config.json`

	Returns:
		BytesIO: zip archive bytes with included `config.json`
	"""

	zip_archive = zipfile.ZipFile(file=file_like_zip, mode='a', compression=zipfile.ZIP_DEFLATED)
	zip_archive.writestr('/config.json', string_to_add)
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


def _get_top_level_and_configs(validate_data: tuple) -> list:
	"""
	Prepares data returned by the recommerce validation function for parsing.
	Should only be used when the validated config was correct

	Args:
		validate_data (tuple): return of recommerce validation function, when config was correct

	Returns:
		list: of tuples, the first tuple value indecates the top level ('hyperparameter' or 'environment')
			and the second value is the corresponding config.
			Length will be between 1 and 2.
	"""
	assert tuple == type(validate_data), \
		f'Data returned by "vaidate_config" for correct config should be tuple, but was {validate_data}'
	result = []
	for config in validate_data:
		if not config:
			continue
		if 'environment' in config:
			result += [('environment', config['environment'])]
		elif 'hyperparameter' in config:
			result += [('hyperparameter', config['hyperparameter'])]
		elif 'rl' in config or 'sim_market' in config:
			# we need to add those two to the same hyperparameter name
			existing_hyperparameter = [item for item in result if 'hyperparameter' in item]
			if existing_hyperparameter:
				new_hyperparameter = ('hyperparameter', {**existing_hyperparameter[0][1], **config})
				result.remove(existing_hyperparameter[0])
				result += [new_hyperparameter]
			else:
				result += [('hyperparameter', config)]
	return result


def _get_config_parts(config_objects: list) -> tuple:
	"""
	Takes a list of tuple with the parsed objects from the config and their top level key
	and returns an 'environment_config' and a 'hyperparameter_config' object to be inserted into the Config object

	Args:
		config_objects (list): list of tuples, first tuple value indecating top-level key ('hyperparameter' / 'environment')
			second value, the actual parsed config object

	Returns:
		tuple: (instance of EnvironmentConfig, instance of HyperparameterConfig)
	"""
	assert len(config_objects) <= 2 and len(config_objects) >= 1, \
		'At least one, at max two config parts should have been parsed'
	environment_config = None
	hyperparameter_config = None
	for top_level, config_part in config_objects:
		if top_level == 'environment':
			environment_config = config_part
		elif top_level == 'hyperparameter':
			hyperparameter_config = config_part
	return environment_config, hyperparameter_config
