import json
import os

from django.http import HttpResponseRedirect  # HttpResponse
from django.shortcuts import render
from django.utils import timezone

from .forms import UploadFileForm
from .handle_uploading import handle_uploaded_file
from .models import Container
from .handle_requests import send_post_request, send_get_request, update_container


def index(request):
	return render(request, 'index.html')


def upload(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		# print(request.FILES, request.POST)
		handle_uploaded_file(request.FILES['upload_config'])
		return HttpResponseRedirect('/start_container')   # render(request, 'start_container.html')
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})


def observe(request):
	if request.method == 'POST':
		if 'health' in request.POST:
			# assuming the id always stays the same
			response = send_get_request('health', request.POST)
			if response:
				update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['health_status']})
		if 'kill' in request.POST:
			response = send_get_request('kill', request.POST)
			if 'killed' in response['health_status']:
				# remove the docker container from the database
				# TODO add a success message for the user
				Container.objects.get(container_id=response['id']).delete()
	all_containers = Container.objects.all()
	return render(request, 'observe.html', {'all_saved_containers': all_containers})


def download(request):
	return render(request, 'download.html')


def start_container(request):
	if request.method == 'POST':
		# the start button was pressed
		config_file = request.POST['filename']
		# read the right config file
		with open(os.path.join('configurations', config_file), 'r') as file:
			config_dict = json.load(file)
			response = send_post_request('start', config_dict)
		if response:
			# TODO add success banner, with new container id
			# put container into database
			Container.objects.create(container_id=response['id'], config_file=config_dict)
			return HttpResponseRedirect('/observe')
		else:
			# TODO tell the user it didnt work
			pass
	file_names = None
	if os.path.exists('configurations'):
		file_names = os.listdir('configurations')
	return render(request, 'start_container.html', {'file_names': file_names})
