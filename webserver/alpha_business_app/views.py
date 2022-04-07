import datetime
from uuid import uuid4

import requests
from django.http import Http404, HttpResponse
from django.shortcuts import render

from recommerce.configuration.config_validation import validate_config

from .buttons import ButtonHandler
from .config_parser import ConfigFlatDictParser
from .forms import UploadFileForm
from .handle_files import handle_uploaded_file
from .handle_requests import DOCKER_API
from .models.config import Config
from .models.container import Container


def detail(request, container_id) -> HttpResponse:
	try:
		wanted_container = Container.objects.get(id=container_id)
	except Container.DoesNotExist as error:
		raise Http404('Container does not exist') from error
	if request.POST.get('action', '') == 'remove':
		# if we  want to remove the container, we need to redirect to a different site
		button_handler = ButtonHandler(request, view='observe.html', container=wanted_container, rendering_method='archived')
	else:
		# otherwise we doa different action and want to stay on the site
		button_handler = ButtonHandler(request, view='details.html', container=wanted_container)
	return button_handler.do_button_click()


def download(request) -> HttpResponse:
	button_handler = ButtonHandler(request, view='download.html')
	return button_handler.do_button_click()


def index(request) -> HttpResponse:
	return render(request, 'index.html')


def observe(request) -> HttpResponse:
	button_handler = ButtonHandler(request, view='observe.html', rendering_method='archived')
	return button_handler.do_button_click()


def upload(request) -> HttpResponse:
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		return handle_uploaded_file(request, request.FILES['upload_config'])
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})


def configurator(request):
	button_handler = ButtonHandler(request, view='configurator.html', rendering_method='config')
	return button_handler.do_button_click()


def delete_config(request, config_id) -> HttpResponse:
	try:
		wanted_config = Config.objects.get(id=config_id)
	except Config.DoesNotExist as error:
		raise Http404('Config does not exist') from error
	button_handler = ButtonHandler(request, view='delete_config.html', wanted_config=wanted_config, rendering_method='config')
	return button_handler.do_button_click()


# AJAX relevant views
def agent(request):
	return render(request, 'configuration_items/agent.html', {'id': str(uuid4())})


def api_availability(request):
	try:
		api_is_available = requests.get(f'{DOCKER_API}/api_health', timeout=1)
	except requests.exceptions.RequestException:
		current_time = datetime.datetime.now().strftime('%H:%M:%S')
		return render(request, 'api_buttons/api_health_button.html', {'api_timeout': f'{current_time}'})

	current_time = datetime.datetime.now().strftime('%H:%M:%S')
	if api_is_available.status_code == 200:
		return render(request, 'api_buttons/api_health_button.html', {'api_success': f'{current_time}'})
	return render(request, 'api_buttons/api_health_button.html', {'api_docker_timeout': f'{current_time}'})


def config_validation(request):
	if request.method == 'POST':
		post_request = request.POST
		print(post_request)
		# convert formdata doct to normal form dict
		resulting_dict = {
			'environment-agents-name': [],
			'environment-agents-agent_class': [],
			'environment-agents-argument': []
		}
		for index in range(0, len(post_request) // 2):
			current_name = post_request[f'formdata[{index}][name]']
			current_value = post_request[f'formdata[{index}][value]']
			print(current_name, current_value, sep='\t')
			if 'agents' in current_name:
				resulting_dict[current_name] += [current_value]
			else:
				resulting_dict[current_name] = [current_value]

		config_dict = ConfigFlatDictParser().flat_dict_to_hierarchical_config_dict(resulting_dict)

		validate_status, validate_data = validate_config(config=config_dict, config_is_final=True)
		if not validate_status:
			return render(request, 'notices/config_valid_field.html', {'config_is_valid': False, 'message': validate_data})
	return render(request, 'notices/config_valid_field.html', {'config_is_valid': True, 'message': 'This config is valid'})
