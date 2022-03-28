import copy

from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone

from .config_merger import ConfigMerger
from .config_parser import ConfigFlatDictParser, ConfigModelParser
from .handle_files import download_file
from .handle_requests import send_get_request, send_get_request_with_streaming, send_post_request, stop_container
from .models.config import Config
from .models.container import Container, update_container


class ButtonHandler():
	def __init__(self,
		request,
		view: str,
		container: Container = None,
		config: Config = None,
		rendering_method: str = 'default',
		data: str = None) -> None:
		"""
		This handler can be used to implement different behaviour when a button is pressed.
		You can add more keywords in `do_button_click()` or implement your own renderings and add them to `_decide_rendering()`.

		Args:
			request (WSGIRequest): post request send to the server when a button is pressed. The keyword 'action' defines what happens.
			view (str): The view that should be rendered when the button action is done.
			container (Container, optional): a container that could be used i.e. for the rendering. Defaults to None.
			rendering_method (str, optional): keyword for the rendering methode, see `_decide_rendering()`. Defaults to 'default'.
			data (str, optional): other data that can be used for rendering, i.e. logs. Defaults to None.
			wanted_config (Config, optional): a config that can be used for any action
		"""
		self.request = request
		self.view_to_render = view
		self.wanted_container = container
		self.rendering_method = rendering_method
		self.data = data
		self.message = [None, None]
		self.wanted_key = None
		self.all_containers = Container.objects.all()
		self.wanted_config = config

		if request.method == 'POST':
			self.wanted_key = request.POST['action']
			if 'container_id' in request.POST:
				wanted_container_id = request.POST['container_id'].strip()
				self.wanted_container = Container.objects.get(id=wanted_container_id)

	def do_button_click(self) -> HttpResponse:
		"""
		Call this function after initializing a ButtonHandler. It will decide, depending on the given keyword in the request,
		which action to perform and return an appropriate rendering for a view.

		Returns:
			HttpResponse: response including the rendering for an appropriate view.
		"""
		if self.wanted_key == 'data':
			return self._download_data()
		if self.wanted_key == 'delete':
			return self._delete_container()
		if self.wanted_key == 'data/tensorboard':
			return self._tensorboard_link()
		if self.wanted_key == 'health':
			return self._health()
		if self.wanted_key == 'pause':
			return self._toggle_pause()
		if self.wanted_key == 'remove':
			return self._remove()
		if self.wanted_key == 'start':
			return self._start()
		if self.wanted_key == 'pre-fill':
			return self._pre_fill()
		if self.wanted_key == 'logs':
			return self._logs()
		if self.wanted_key == 'manage_config':
			return self._manage_config()
		# no button was clicked?
		return self._decide_rendering()

	# PRIVATE METHODS
	def _decide_rendering(self) -> HttpResponse:
		"""
		Will return you a rendering depending on the given keyword in `self.rendering_method`.
		Extend this function if you want more rendering possibilities.

		Returns:
			HttpResponse: a defined rendering
		"""
		if self.rendering_method == 'default':
			return self._render_default()
		elif self.rendering_method == 'config':
			return self._render_configuration()
		return self._render_without_archived()

	def _default_params_for_view(self) -> dict:
		"""
		This will give you all parameters, that might be in any view

		Returns:
			dict: a dictionary of all_containers, a container, data and the error or success message.
		"""
		return {'all_saved_containers': self.all_containers,
				'container': self.wanted_container,
				'data': self.data,
				**self._message_for_view()}

	def _message_for_view(self) -> dict:
		"""
		Will return the message and their value in dict for the template

		Returns:
			dict: contains the state of the message (i.e. success or error) and the message itself.
		"""
		return {self.message[0]: self.message[1]}

	def _params_for_config(self) -> dict:
		"""
		Will return all parameters necessary for the configurator.

		Returns:
			dict: contains all current configuration objects, the current config and this config as dict if it exists.
		"""
		return {'all_configurations': Config.objects.all(),
			'config': self.wanted_config,
			'config_dict': self.wanted_config.as_dict() if self.wanted_config else None}

	def _render_default(self) -> HttpResponse:
		"""
		This will return a rendering for `self.view` with the default parameters.

		Returns:
			HttpResponse: a default rendering with default values, some might be set by the different functions.
		"""
		self.all_containers = Container.objects.all()
		return render(self.request, self.view_to_render, self._default_params_for_view())

	def _render_without_archived(self) -> HttpResponse:
		"""
		This will return a rendering for `self.view` without all 'archived' containers.

		Returns:
			HttpResponse: a default rendering without all archived containers.
		"""
		self.all_containers = Container.objects.all().exclude(health_status='archived')
		return render(self.request, self.view_to_render, self._default_params_for_view())

	def _render_configuration(self) -> HttpResponse:
		"""
		This will return a rendering for `self.view` with params for config and the given message

		Returns:
			HttpResponse: a rendering for the configurator with all configuration parameters.
		"""
		return render(self.request, self.view_to_render, {**self._params_for_config(), **self._message_for_view()})

	def _delete_container(self) -> HttpResponse:
		"""
		This will delete the selected container and all data belonging to this container.

		Returns:
			HttpResponse: a defined rendering.
		"""
		raw_data = {'container_id': self.wanted_container.id}
		if not self.wanted_container.is_archived():
			self.message = stop_container(raw_data).status()

		if self.message[0] == 'success' or self.wanted_container.is_archived():
			self.wanted_container.delete()
			self.wanted_container = None
			self.message = ['success', 'You successfully removed all data']
		return self._decide_rendering()

	def _download_data(self) -> HttpResponse:
		"""
		This will send an API  request to get the latest data of the container.
		Will return an error in the response if the container is archived.

		Returns:
			HttpResponse: The latest data from the container, or a response containing the error field.
		"""
		if self.wanted_container.is_archived():
			self.message = ['error', 'You cannot download data from archived containers']
			return self._decide_rendering()
		else:
			response = send_get_request_with_streaming('data', self.wanted_container.id)
			if response.ok():
				# save data from api and make it available for the user
				return download_file(response.content, self.request.POST['file_type'] == 'zip', self.wanted_container)
			else:
				self.message = response.status()
				return self._decide_rendering()

	def _health(self) -> HttpResponse:
		"""
		This will send an API request to get the health status of a container and updates the container in the database.

		Returns:
			HttpResponse: A default response with default values or a response containing the error field.
		"""
		response = send_get_request('health', self.request.POST)
		if response.ok():
			response = response.content
			update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
		else:
			self.message = response.status()
		self.wanted_container = Container.objects.get(id=self.wanted_container.id)

		return self._decide_rendering()

	def _logs(self) -> HttpResponse:
		"""
		This will send an API request to get the logs of the container.
		They will be stored in `self.data` in order to render them.

		Returns:
			HttpResponse: A default response with default values or a response containing the error field.
		"""
		response = send_get_request('logs', self.request.POST)
		self.data = ''
		if response.ok():
			# reverse the output for better readability
			self.data = response.content['data'].splitlines()
			self.data.reverse()
			self.data = '\n'.join(self.data)
		return self._decide_rendering()

	def _manage_config(self) -> HttpResponse:
		"""
		This should be called whenever a configuration should be managed from UI.
		If there is a 'delete' in the post, self.wanted_config will be deleted.

		Returns:
			HttpResponse: if the action on the config was successful, it will redirect you to 'configurator'
		"""
		if 'delete' in self.request.POST:
			self.wanted_config.delete()
		return redirect('/configurator')

	def _pre_fill(self) -> HttpResponse:
		"""
		This function will be called when the config form should be prefilled with values from the config.
		It converts a list of given config objects to dicts and merges these dicts.
		The merged result and the errors which came up when merging will be passed to the view.

		Returns:
			HttpResponse: a rendering for the view with the prefill dict and an error dict.
		"""
		post_request = dict(self.request.POST.lists())
		if 'config_id' not in post_request:
			return self._decide_rendering()
		merger = ConfigMerger()
		final_dict, error_dict = merger.merge_config_objects(post_request['config_id'])
		return render(self.request, self.view_to_render,
			{'prefill': final_dict, 'error_dict': error_dict, 'all_configurations': Config.objects.all()})

	def _remove(self) -> HttpResponse:
		"""
		This will send an API request to stop and remove the selected container.

		Returns:
			HttpResponse: An appropriate rendering
		"""
		self.message = stop_container(self.request.POST).status()
		return self._decide_rendering()

	def _start(self) -> HttpResponse:
		"""
		Sends a post request to the API with a config in the body to start the container.

		Returns:
			HttpResponse: An appropriate rendering, or a redirect to the `observe` view.
		"""
		# convert post request to normal dict
		post_request = dict(self.request.POST.lists())

		# TODO: error, when multiple agents have the same name!
		flat_parser = ConfigFlatDictParser()
		config_dict = flat_parser.flat_dict_to_hierarchical_config_dict(post_request)
		# TODO: assert config dict is valid
		response = send_post_request('start', config_dict)
		if response.ok():
			# put container into database
			response = response.content
			# check if a container with the same id already exists
			if Container.objects.filter(id=response['id']).exists():
				# we will kindly ask the user to try it again and stop the container
				# TODO insert better handling here
				print('the new container has the same id, as another container')
				self.message = ['error', 'please try again']
				return self._remove()
			print('after post request')
			# get all necessary parameters for container object
			container_name = self.request.POST['experiment_name']
			container_name = container_name if container_name != '' else response['id'][:10]
			parser = ConfigModelParser()
			config_object = parser.parse_config(copy.deepcopy(config_dict))
			command = config_object.environment.task
			Container.objects.create(id=response['id'], config=config_object, name=container_name, command=command)
			config_object.name = f'used for {container_name}'
			config_object.save()
			return redirect('/observe', {'success': 'You successfully launched an experiment'})
		else:
			self.message = response.status()
			return self._decide_rendering()

	def _tensorboard_link(self) -> HttpResponse:
		"""
		This will send an API request to get the tensorboard link for the selected container.
		It will used a cached tensorboard link if available.

		Returns:
			HttpResponse: An appropriate rendering, or a redirect to the tensorboard.
		"""
		if self.wanted_container.has_tensorboard_link():
			return redirect(self.wanted_container.tensorboard_link)
		response = send_get_request('data/tensorboard', self.request.POST)
		if response.ok():
			update_container(self.wanted_container.id, {'tensorboard_link': response.content['data']})
			return redirect(response.content['data'])
		else:
			self.message = response.status()
			return self._decide_rendering()

	def _toggle_pause(self) -> HttpResponse:
		"""
		This will send an API request to pause/unpause the currently running container.

		Returns:
			HttpResponse: A default response with default values or a response containing the error field.
		"""
		# check, whether the request wants to pause or to unpause the container
		if self.wanted_container.is_paused():
			response = send_get_request('unpause', self.request.POST)
		else:
			response = send_get_request('pause', self.request.POST)

		if response.ok():
			response = response.content
			update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
		else:
			self.message = response.status()
		self.wanted_container = Container.objects.get(id=self.wanted_container.id)

		return self._decide_rendering()
