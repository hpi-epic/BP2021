import json
import os

from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone

from .handle_files import download_file
from .handle_requests import send_get_request, send_get_request_with_streaming, send_post_request, stop_container
from .models import Container, update_container


class ButtonHandler():
	def __init__(self, request, view: str, container: Container = None, rendering_method: str = 'default', data: str = None) -> None:
		"""
		This handler can be used to implement different behaviour when a button is pressed.
		You can add more keywords in `do_button_click()` or implement your own renderings and add them to `_decide_rendering()`.

		Args:
			request (WSGIRequest): post request send to the server when a button is pressed. The keyword 'action' defines what happens.
			view (str): The view that should be rendered when the button action is done.
			container (Container, optional): a container that could be used i.e. for the rendering. Defaults to None.
			rendering_method (str, optional): keyword for the rendering methode, see `_decide_rendering()`. Defaults to 'default'.
			data (str, optional): other data that can be used for rendering, i.e. logs. Defaults to None.
		"""
		self.request = request
		self.view_to_render = view
		self.wanted_container = container
		self.rendering_method = rendering_method
		self.data = data
		self.message = [None, None]
		self.wanted_key = None
		self.all_containers = Container.objects.all()

		if request.method == 'POST':
			self.wanted_key = request.POST['action']
			if 'container_id' in request.POST:
				wanted_container_id = request.POST['container_id'].strip()
				self.wanted_container = Container.objects.get(container_id=wanted_container_id)

	def do_button_click(self) -> HttpResponse:
		"""
		Call this function after initializing a ButtonHandler. It will decide, depending on the given keyword in the request,
		which action to perform and return an appropriate rendering for a view.

		Returns:
			HttpResponse: response including the rendering for an appropriate view.
		"""
		if 'data' == self.wanted_key:
			return self._download_data()
		if 'delete' == self.wanted_key:
			return self._delete_container()
		if 'data/tensorboard' == self.wanted_key:
			return self._tensorboard_link()
		if 'health' == self.wanted_key:
			return self._health()
		if 'remove' == self.wanted_key:
			return self._remove()
		if 'start' == self.wanted_key:
			return self._start()
		if 'logs' == self.wanted_key:
			return self._logs()
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
		elif self.rendering_method == 'files':
			return self._render_files()
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
				self.message[0]: self.message[1]}

	def _render_default(self) -> HttpResponse:
		"""
		This will return a rendering for `self.view` with the default parameters.

		Returns:
			HttpResponse: A default rendering with default values, some might be set by the different functions.
		"""
		self.all_containers = Container.objects.all()
		return render(self.request, self.view_to_render, self._default_params_for_view())

	def _render_without_archived(self) -> HttpResponse:
		"""
		This will return a rendering for `self.view` without all 'archived' containers.

		Returns:
			HttpResponse: A default rendering with default values, some might be set by the different functions.
		"""
		self.all_containers = Container.objects.all().exclude(health_status='archived')
		return render(self.request, self.view_to_render, self._default_params_for_view())

	def _render_files(self):
		"""
		This will return a rendering for `self.view` with all `file_names` from `configurations`.

		Returns:
			HttpResponse: A rendering with all file names and the error or success message.
		"""
		file_names = None
		if os.path.exists('configurations'):
			file_names = os.listdir('configurations')
		return render(self.request, self.view_to_render, {'file_names': file_names, self.message[0]: self.message[1]})

	# BUTTON ACTION HANDLING
	def _delete_container(self) -> HttpResponse:
		"""
		This will delete the selected container and all data belonging to this container.

		Returns:
			HttpResponse: a defined rendering
		"""
		raw_data = {'container_id': self.wanted_container.id()}
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
			self.message = ['error', 'You cannot downoload data from archived containers']
			return self._decide_rendering()
		else:
			response = send_get_request_with_streaming('data', self.wanted_container.id())
			if response.ok():
				# save data from api and make it available for the user
				response = response.content
				return download_file(response, self.request.POST['file_type'] == 'zip')
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
		self.wanted_container = Container.objects.get(container_id=self.wanted_container.id())
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
		post_request = self.request.POST

		requested_command = post_request['command_selection']
		# the start button was pressed
		config_file = post_request['filename']
		# read the right config file
		with open(os.path.join('configurations', config_file), 'r') as file:
			config_dict = json.load(file)
			response = send_post_request('start', config_dict, requested_command)

		if response.ok():
			# put container into database
			response = response.content
			# check if a container with the same id already exists
			if Container.objects.filter(container_id=response['id']).exists():
				# we will kindly ask the user to try it again and stop the container
				# TODO insert better handling here
				print('the new container has the same id, as another container')
				self.message = ['error', 'please try again']
				return self._remove()
			container_name = self.request.POST['experiment_name']
			container_name = container_name if container_name != '' else response['id'][:10]
			Container.objects.create(container_id=response['id'], config_file=config_dict, name=container_name, command=requested_command)
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
			update_container(self.wanted_container.id(), {'tensorboard_link': response.content['data']})
			return redirect(response.content['data'])
		else:
			self.message = response.status()
			return self._decide_rendering()
