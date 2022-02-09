from django.http import Http404, HttpResponseRedirect
from django.shortcuts import render

from .buttons import ButtonHandler
from .forms import UploadFileForm
from .handle_files import handle_uploaded_file
from .models import Container


def detail(request, container_id):
	try:
		wanted_container = Container.objects.get(container_id=container_id)
	except Container.DoesNotExist:
		raise Http404('Container does not exist')
	button_handler = ButtonHandler(request, view='details.html', container=wanted_container)
	return button_handler.do_button_click()


def download(request):
	button_handler = ButtonHandler(request, view='download.html')
	return button_handler.do_button_click()


def index(request):
	return render(request, 'index.html')


def observe(request):
	button_handler = ButtonHandler(request, view='observe.html', rendering_method='archived')
	return button_handler.do_button_click()


def start_container(request):
	button_handler = ButtonHandler(request, view='start_container.html', rendering_method='files')
	return button_handler.do_button_click()


def upload(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		handle_uploaded_file(request.FILES['upload_config'])
		return HttpResponseRedirect('/start_container', {'success': 'You successfully uploaded a config file'})
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})
