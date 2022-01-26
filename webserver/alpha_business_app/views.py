import json
import os

import requests
from django.http import HttpResponseRedirect  # HttpResponse
from django.shortcuts import render
from django.utils import timezone

from .forms import UploadFileForm
from .handle_uploading import handle_uploaded_file
from .models import Container

# start api with uvicorn app:app --reload
DOCKER_API = 'http://127.0.0.1:8000'


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
	if request.method == 'POST' and 'health_check' in request.POST:
		wanted_container = request.POST['health_check']
		response = requests.get(DOCKER_API + '/health', params={'id': str(wanted_container)})
		response = response.json()
		saved_container = Container.objects.get(container_id=wanted_container)
		saved_container.last_check_at = timezone.now()
		saved_container.health_status = response['health_status']
		saved_container.save()
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
			response = requests.post(DOCKER_API + '/start', json=config_dict)
		if response.ok:
			# TODO add success banner, with new container id
			# put container into database
			response_json = response.json()
			Container.objects.create(container_id=response_json['id'], config_file=config_dict)
			return HttpResponseRedirect('/observe')
		else:
			# TODO tell the user it didnt work
			pass
	files_names = os.listdir('configurations')
	return render(request, 'start_container.html', {'file_names': files_names})
