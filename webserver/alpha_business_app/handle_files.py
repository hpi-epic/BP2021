import json
import tarfile
import zipfile
from io import BytesIO

from django.http import HttpResponse
from django.shortcuts import redirect, render

from .config_parser import ConfigModelParser
from .models.config import *
from .models.container import Container
from .validation import check_dict_keys


# https://stackoverflow.com/questions/14902299/json-loads-allows-duplicate-keys-in-a-dictionary-overwriting-the-first-value
def _dict_raise_on_duplicates(ordered_pairs):
	"""
	Reject duplicate keys.
	"""
	d = {}
	for k, v in ordered_pairs:
		if k in d:
			raise ValueError('Your config contains duplicate keys: %r' % (k,))
		else:
			d[k] = v
	return d


def handle_uploaded_file(request, uploaded_config) -> HttpResponse:
	"""
	Checks if an uploaded config file is valid and parses it to the datastructure.

	Args:
		request (Request):
		uploaded_config (InMemoryUploadedFile): by user uploaded config file

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

	# figure out which parts of the config file belong to hyperparameter or environment config
	hyperparameter_fields = get_config_field_names(HyperparameterConfig)
	environment_fields = get_config_field_names(EnvironmentConfig)

	# figure out which keywords belong to hyperparameter and which keywords belong to environment
	# TODO: should be outsourced to check
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

	parser = ConfigModelParser()
	hyperparameter_config = None
	environment_config = None
	# TODO: insert actual checking of config here. This is very hacky
	try:
		if contains_hyperparameter is True:
			hyperparameter_config = parser.parse_config_dict_to_datastructure('hyperparameter', hyperparameter_configs)
		if contains_environment is True:
			environment_config = parser.parse_config_dict_to_datastructure('environment', environment_configs)
	except ValueError:
		return render(request, 'upload.html', {'error': 'Your config is wrong'})

	Config.objects.create(environment=environment_config, hyperparameter=hyperparameter_config, name=request.POST['config_name'])
	return redirect('/configurator', {'success': 'You successfully uploaded a config file'})


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
