import json
import os

from django.http import HttpResponse, HttpResponseRedirect  # HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone

from .forms import UploadFileForm
from .handle_files import archive_files, download_file, handle_uploaded_file, save_data
from .handle_requests import send_get_request, send_get_request_with_streaming, send_post_request, stop_container
from .models import Container, update_container


def detail(request, container_id):
	return HttpResponse('You are looking at container %s.' % container_id)


def download(request):
	message = [None, None]
	all_containers = Container.objects.all()
	if request.method == 'POST':
		all_keys = list(request.POST.keys())
		all_keys.remove('csrfmiddlewaretoken')
		assert 1 == len(all_keys), 'You can only use one request at a time'
		wanted_key = all_keys[0]
		wanted_container_id = request.POST[wanted_key]
		wanted_container = Container.objects.get(container_id=wanted_container_id)

		if 'data-all' == wanted_key:
			# the user wants the all data we saved
			if not wanted_container.has_data():
				return render(request, 'download.html',
					{'all_saved_containers': all_containers,
					'error': 'You have not yet saved any data belonging to this container'})
			return archive_files(wanted_container_id)

		if 'data-latest' == wanted_key:
			# the user only wants the lates data
			if wanted_container.is_archived():
				message = ['error', 'You cannot downoload data from archived containers']
			else:
				response = send_get_request_with_streaming('data', wanted_container_id)
				if response.ok():
					# save data from api and make it available for the user
					response = response.content()
					path = save_data(response, wanted_container_id)
					return download_file(path)
				else:
					message = response.status()

		if 'remove' == wanted_key:
			if not wanted_container.is_archived():
				message = stop_container(request.POST).status()

			if message[0] == 'success' or wanted_container.is_archived():
				wanted_container.delete()
				all_containers = Container.objects.all()
				message = ['success', 'You successfully removed all data']
	return render(request, 'download.html', {'all_saved_containers': all_containers, message[0]: message[1]})


def index(request):
	return render(request, 'index.html')


def observe(request):
	message = [None, None]
	if request.method == 'POST':
		if 'data/tensorboard' in request.POST:
			response = send_get_request('data/tensorboard', request.POST)
			if response.ok():
				return redirect(response.content()['data'])
			else:
				message = response.status()

		if 'health' in request.POST:
			response = send_get_request('health', request.POST)
			if response.ok():
				response = response.content()
				update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
			else:
				message = response.status()

		if 'remove' in request.POST:
			message = stop_container(request.POST).status()

	all_containers = Container.objects.all().exclude(health_status='archived')
	return render(request, 'observe.html', {'all_saved_containers': all_containers, message[0]: message[1]})


def start_container(request):
	message = [None, None]
	if request.method == 'POST':
		# the start button was pressed
		config_file = request.POST['filename']
		# read the right config file
		with open(os.path.join('configurations', config_file), 'r') as file:
			config_dict = json.load(file)
			response = send_post_request('start', config_dict, request.POST['command_selection'])
		print(response.ok())
		if response.ok():
			# put container into database
			response = response.content()
			container_name = request.POST['experiment_name']
			container_name = container_name if container_name != '' else response['id']
			Container.objects.create(container_id=response['id'], config_file=config_dict, name=container_name)
			return redirect('/observe', {'success': 'You successfully launched an experiment'})
		else:
			message = response.status()
	file_names = None
	if os.path.exists('configurations'):
		file_names = os.listdir('configurations')
	return render(request, 'start_container.html', {'file_names': file_names, message[0]: message[1]})


def upload(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		handle_uploaded_file(request.FILES['upload_config'])
		return HttpResponseRedirect('/start_container', {'success': 'You successfully uploaded a config file'})
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})
