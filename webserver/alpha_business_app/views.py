from uuid import uuid4

from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse
from django.shortcuts import render

from recommerce.configuration.config_validation import validate_config

from .buttons import ButtonHandler
from .config_parser import ConfigFlatDictParser
from .forms import UploadFileForm
from .handle_files import handle_uploaded_file
from .handle_requests import get_api_status
from .models.config import Config
from .models.container import Container
from .selection_manager import SelectionManager

selection_manager = SelectionManager()


@login_required
def detail(request, container_id) -> HttpResponse:
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	try:
		wanted_container = Container.objects.get(id=container_id, user=request.user)
	except Container.DoesNotExist as error:
		raise Http404('Container does not exist') from error
	if request.POST.get('action', '') == 'remove':
		# if we  want to remove the container, we need to redirect to a different site
		button_handler = ButtonHandler(request, view='observe.html', container=wanted_container, rendering_method='archived')
	else:
		# otherwise we doa different action and want to stay on the site
		button_handler = ButtonHandler(request, view='details.html', container=wanted_container)
	return button_handler.do_button_click()


@login_required
def download(request) -> HttpResponse:
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	button_handler = ButtonHandler(request, view='download.html')
	return button_handler.do_button_click()


@login_required
def index(request) -> HttpResponse:
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	return render(request, 'index.html')


@login_required
def observe(request) -> HttpResponse:
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	button_handler = ButtonHandler(request, view='observe.html', rendering_method='archived')
	return button_handler.do_button_click()


@login_required
def upload(request) -> HttpResponse:
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		if not request.FILES:
			return render(request, 'upload.html', {'form': form, 'error': 'You need to upload a file before submitting'})
		return handle_uploaded_file(request, request.FILES['upload_config'])
	else:
		form = UploadFileForm()
	return render(request, 'upload.html', {'form': form})


@login_required
def configurator(request):
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	button_handler = ButtonHandler(request, view='configurator.html', rendering_method='config')
	return button_handler.do_button_click()


@login_required
def delete_config(request, config_id) -> HttpResponse:
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	try:
		wanted_config = Config.objects.get(id=config_id)
	except Config.DoesNotExist as error:
		raise Http404('Config does not exist') from error
	button_handler = ButtonHandler(request, view='delete_config.html', wanted_config=wanted_config, rendering_method='config')
	return button_handler.do_button_click()


# AJAX relevant views
@login_required
def agent(request):
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	return render(request, 'configuration_items/agent.html', {'id': str(uuid4()), 'name': 'Competitor'})


def api_availability(request):
	if not request.user.is_authenticated:
		return render(request, 'api_buttons/api_health_button.html')
	parameter_dict = get_api_status()
	return render(request, 'api_buttons/api_health_button.html', parameter_dict)


@login_required
def config_validation(request):
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	if request.method == 'POST':
		post_request = request.POST
		# convert formdata dict to normal form dict
		resulting_dict = {
			'environment-agents-name': [],
			'environment-agents-agent_class': [],
			'environment-agents-argument': []
		}
		for index in range(len(post_request) // 2):
			current_name = post_request[f'formdata[{index}][name]']
			current_value = post_request[f'formdata[{index}][value]']
			if 'agents' in current_name:
				resulting_dict[current_name] += [current_value]
			else:
				resulting_dict[current_name] = [current_value]

		config_dict = ConfigFlatDictParser().flat_dict_to_hierarchical_config_dict(resulting_dict)

		validate_status, validate_data = validate_config(config=config_dict, config_is_final=True)
		if not validate_status:
			return render(request, 'notice_field.html', {'error': validate_data})
	return render(request, 'notice_field.html', {'success': 'This config is valid'})


def marketplace_changed(request):
	if not request.user.is_authenticated:
		return HttpResponse('Unauthorized', status=401)
	marketplace_class = None
	if request.method == 'POST':
		post_request = request.POST
		# print(post_request)
		marketplace_class = post_request['marketplace']
		raw_html = post_request['agents_html']
	return HttpResponse(content=SelectionManager().get_correct_agents_html_on_marketplace_change(request, marketplace_class, raw_html))
