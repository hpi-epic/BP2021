import json
import os

from django.http import Http404, HttpResponseRedirect
from django.shortcuts import redirect, render

from .buttons import ButtonHandler
from .forms import UploadFileForm
from .handle_files import handle_uploaded_file
from .handle_requests import send_post_request
from .models import Container


def detail(request, container_id):
	try:
		wanted_container = Container.objects.get(container_id=container_id)
	except Container.DoesNotExist:
		raise Http404('Container does not exist')
	button_handler = ButtonHandler(request, view='details.html', container=wanted_container)
	return button_handler.do_button_click()


def download(request):
	button_handler = ButtonHandler(request, 'download.html')
	return button_handler.do_button_click()


def index(request):
	return render(request, 'index.html')


def observe(request):
	button_handler = ButtonHandler(request, 'observe.html')
	return button_handler.do_button_click()


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
