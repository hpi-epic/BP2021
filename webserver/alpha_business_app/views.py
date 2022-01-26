import json
import os

import requests
from django.http import HttpResponseRedirect  # HttpResponse
from django.shortcuts import render

from .forms import UploadFileForm
from .handle_uploading import handle_uploaded_file

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
	response = requests.get(DOCKER_API + '/health', params={'id': '1'})
	response = response.json()
	print(response['is_alive'])
	return render(request, 'observe.html', {'health': 'hi'})


def download(request):
	return render(request, 'download.html')


def start_container(request):
	if request.method == 'POST':
		print(request.POST)  # dir(request))
		config_file = request.POST['filename']
		with open(os.path.join('configurations', config_file), 'r') as file:
			config_dict = json.load(file)
			response = requests.post(DOCKER_API + '/start', json=config_dict)
		if response.ok:
			# put container into database
			print(response.status_code)
			print(response.json())
	files_names = os.listdir('configurations')
	return render(request, 'start_container.html', {'file_names': files_names})
