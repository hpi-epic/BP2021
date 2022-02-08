import json
import os

from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone

from .handle_files import archive_files, download_file, save_data
from .handle_requests import send_get_request, send_get_request_with_streaming, send_post_request, stop_container
from .models import Container, update_container


class ButtonHandler():
	def __init__(self, request, view: str, container: Container = None, rendering_method: str = 'default') -> None:
		self.request = request
		self.view_to_render = view
		self.message = [None, None]
		self.wanted_container = container
		self.wanted_key = None
		self.all_containers = Container.objects.all()
		self.rendering_method = rendering_method

		if request.method == 'POST':
			all_keys = list(self.request.POST.keys())
			all_keys.remove('csrfmiddlewaretoken')
			if len(all_keys) != 1:
				# we have multiple parameters for this request, assuming want to start a container
				self.wanted_key = 'start'
			else:
				self.wanted_key = all_keys[0]
				self.wanted_container_id = self.request.POST[self.wanted_key]
				self.wanted_container = Container.objects.get(container_id=self.wanted_container_id)

	def do_button_click(self):
		if 'data-all' == self.wanted_key:
			return self._download_all_data()
		if 'data-latest' == self.wanted_key:
			return self._download_latest_data()
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
		return self._decide_rendering()

	def _decide_rendering(self) -> HttpResponse:
		if self.rendering_method == 'default':
			return self._render_default()
		elif self.rendering_method == 'files':
			return self._render_files()
		return self._render_without_archived()

	def _default_params_for_view(self) -> dict:
		return {'all_saved_containers': self.all_containers,
				'container': self.wanted_container,
				self.message[0]: self.message[1]}

	def _delete_container(self):
		raw_data = {'remove': self.request.POST['delete']}
		if not self.wanted_container.is_archived():
			self.message = stop_container(raw_data).status()

		if self.message[0] == 'success' or self.wanted_container.is_archived():
			self.wanted_container.delete()
			self.all_containers = Container.objects.all()
			self.message = ['success', 'You successfully removed all data']
		return self._decide_rendering()

	def _download_all_data(self):
		# the user wants the all data we saved
		if not self.wanted_container.has_data():
			self.message = ['error', 'You have not yet saved any data belonging to this container']
			return self._decide_rendering()
		return archive_files(self.wanted_container_id)

	def _download_latest_data(self):
		# the user only wants the lates data
		if self.wanted_container.is_archived():
			self.message = ['error', 'You cannot downoload data from archived containers']
		else:
			response = send_get_request_with_streaming('data', self.wanted_container_id)
			if response.ok():
				# save data from api and make it available for the user
				response = response.content()
				path = save_data(response, self.wanted_container_id)
				return download_file(path)
			else:
				self.message = response.status()
				return self._decide_rendering()

	def _health(self):
		response = send_get_request('health', self.request.POST)
		if response.ok():
			response = response.content()
			update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
		else:
			self.message = response.status()
		return self._decide_rendering()

	def _render_default(self) -> HttpResponse:
		return render(self.request, self.view_to_render, self._default_params_for_view())

	def _render_without_archived(self) -> HttpResponse:
		self.all_containers = Container.objects.all().exclude(health_status='archived')
		return self._render_default()

	def _render_files(self):
		if os.path.exists('configurations'):
			file_names = os.listdir('configurations')
		return render(self.request, self.view_to_render, {'file_names': file_names, self.message[0]: self.message[1]})

	def _remove(self):
		self.message = stop_container(self.request.POST).status()
		# update all_containers in order to only show non archived containers
		return self._decide_rendering()

	def _start(self):
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
			response = response.content()
			container_name = self.request.POST['experiment_name']
			container_name = container_name if container_name != '' else response['id']
			Container.objects.create(container_id=response['id'], config_file=config_dict, name=container_name, command=requested_command)
			return redirect('/observe', {'success': 'You successfully launched an experiment'})
		else:
			self.message = response.status()
			return self._decide_rendering()

	def _tensorboard_link(self):
		if self.wanted_container.has_tensorboard_link():
			return redirect(self.wanted_container.tensorboard_link)
		response = send_get_request('data/tensorboard', self.request.POST)
		if response.ok():
			update_container(self.wanted_container_id, {'tensorboard_link': response.content()['data']})
			return redirect(response.content()['data'])
		else:
			self.message = response.status()
			return self._decide_rendering()
