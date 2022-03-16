from django.http import Http404, HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from .buttons import ButtonHandler
from .forms import UploadFileForm
from .handle_files import handle_uploaded_file
from .models.container import Container


def detail(request, container_id) -> HttpResponse:
	try:
		wanted_container = Container.objects.get(id=container_id)
	except Container.DoesNotExist as error:
		raise Http404('Container does not exist') from error
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


def start_container(request) -> HttpResponse:
	button_handler = ButtonHandler(request, view='start_container.html', rendering_method='files')
	return button_handler.do_button_click()


def upload(request) -> HttpResponse:
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		handle_uploaded_file(request.FILES['upload_config'])
		return HttpResponseRedirect('/start_container', {'success': 'You successfully uploaded a config file'})
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})
