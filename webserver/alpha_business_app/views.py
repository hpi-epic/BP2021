from django.http import Http404, HttpResponse
from django.shortcuts import render

from .buttons import ButtonHandler
from .forms import UploadFileForm
from .handle_files import handle_uploaded_file
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


def start_container(request) -> HttpResponse:
	button_handler = ButtonHandler(request, view='start_container.html', rendering_method='files')
	return button_handler.do_button_click()


def upload(request) -> HttpResponse:
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		return handle_uploaded_file(request, request.FILES['upload_config'])
		# return HttpResponseRedirect('/start_container', {'success': 'You successfully uploaded a config file'})
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})


def configurator(request):
	if request.method == 'POST':
		print(request.POST)
	return render(request, 'configurator.html', {'all_configurations': Config.objects.all()})
