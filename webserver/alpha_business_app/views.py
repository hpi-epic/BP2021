import json
import os

from django.http import HttpResponseRedirect  # HttpResponse
from django.shortcuts import render
from django.utils import timezone

from .forms import UploadFileForm
from .handle_files import archive_files, download_file, handle_uploaded_file, has_downloaded_data, save_data
from .handle_requests import send_get_request, send_get_request_with_streaming, send_post_request, stop_container
from .models import Container, update_container


def index(request):
	return render(request, 'index.html')


def upload(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		handle_uploaded_file(request.FILES['upload_config'])
		return HttpResponseRedirect('/start_container')
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})


def observe(request):
	if request.method == 'POST':
		if 'health' in request.POST:
			response = send_get_request('health', request.POST)
			if response:
				update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
		if 'remove' in request.POST:
			if not stop_container(request.POST):
				pass
	all_containers = Container.objects.all().exclude(health_status='archived')
	return render(request, 'observe.html', {'all_saved_containers': all_containers})


def download(request):
	all_containers = Container.objects.all()
	if request.method == 'POST':
		if 'data-latest' in request.POST:
			wanted_container = request.POST['data-latest']
			response = send_get_request_with_streaming('data', wanted_container)
			if response:
				# save data from api and make it available for the user
				path = save_data(response, wanted_container)
				return download_file(path)
		if 'data-all' in request.POST:
			wanted_container = request.POST['data-all']
			if not has_downloaded_data(wanted_container):
				return render(request, 'download.html', {'all_saved_containers': all_containers})
			return archive_files(wanted_container)
		if 'remove' in request.POST:
			wanted_container = request.POST['remove']
			if Container.objects.get(container_id=wanted_container).health_status != 'archived':
				stop_container(request.POST)
			Container.objects.get(container_id=wanted_container).delete()
			all_containers = Container.objects.all()

	return render(request, 'download.html', {'all_saved_containers': all_containers})


def start_container(request):
	print(request.POST)
	if request.method == 'POST':
		# the start button was pressed
		config_file = request.POST['filename']
		# read the right config file
		with open(os.path.join('configurations', config_file), 'r') as file:
			config_dict = json.load(file)
			response = send_post_request('start', config_dict, request.POST['command_selection'])
		if response:
			# TODO add success banner, with new container id
			# put container into database
			container_name = request.POST['experiment_name']
			container_name = container_name if container_name != '' else response['id']
			Container.objects.create(container_id=response['id'], config_file=config_dict, name=container_name)
			return HttpResponseRedirect('/observe')
		else:
			# TODO tell the user it didnt work
			pass
	file_names = None
	if os.path.exists('configurations'):
		file_names = os.listdir('configurations')
	return render(request, 'start_container.html', {'file_names': file_names})


def tensorboard(request):
	if (request.GET.get('observe')):
		# mypythoncode.mypythonfunction(int(request.GET.get('mytextbox')))
		# response = send_get_request('data/tensorboard', request.POST)
		# Container.objects.get(container_id=response['id'])
		response = send_get_request('health', request.POST)
		if response:
			update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
	all_containers = Container.objects.all()
	return render(request, 'observe.html', {'all_saved_containers': all_containers})
