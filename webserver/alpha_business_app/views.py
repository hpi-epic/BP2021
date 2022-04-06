import datetime
from uuid import uuid4

import requests
from django.http import Http404, HttpResponse
from django.shortcuts import render

from .buttons import ButtonHandler
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
		return render(request, 'api_buttons/api_health_button.html', {'api_timeout': f'unavailable {current_time}'})

	current_time = datetime.datetime.now().strftime('%H:%M:%S')
	if api_is_available.status_code == 200:
		return render(request, 'api_buttons/api_health_button.html', {'api_success': f'available {current_time}'})
	return render(request, 'api_buttons/api_health_button.html', {'api_docker_timeout': f'unavailable {current_time}'})
