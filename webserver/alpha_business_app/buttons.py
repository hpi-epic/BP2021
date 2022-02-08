from django.shortcuts import redirect, render
from django.utils import timezone

from .handle_files import archive_files, download_file, save_data
from .handle_requests import send_get_request, send_get_request_with_streaming, stop_container
from .models import Container, update_container


class ButtonHandler:
	def __init__(self, request, view: str, container: Container = None) -> None:
		self.request = request
		self.view_to_render = view
		self.message = [None, None]
		self.all_containers = Container.objects.all()
		self.wanted_container = container
		self.wanted_key = None

		if request.method == 'POST':
			all_keys = list(self.request.POST.keys())
			all_keys.remove('csrfmiddlewaretoken')
			assert 1 == len(all_keys), 'You can only use one request at a time'
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
		return self._default_render()

	def _default_params_for_view(self):
		print(self.wanted_container)
		return {'all_saved_containers': self.all_containers,
				'container': self.wanted_container,
				self.message[0]: self.message[1]}

	def _default_render(self):
		return render(self.request, self.view_to_render, self._default_params_for_view())

	def _delete_container(self):
		raw_data = {'remove': self.request.POST['delete']}
		if not self.wanted_container.is_archived():
			self.message = stop_container(raw_data).status()

		if self.message[0] == 'success' or self.wanted_container.is_archived():
			self.wanted_container.delete()
			self.all_containers = Container.objects.all()
			self.message = ['success', 'You successfully removed all data']
		return self._default_render()

	def _download_all_data(self):
		# the user wants the all data we saved
		if not self.wanted_container.has_data():
			self.message = ['error', 'You have not yet saved any data belonging to this container']
			return self._default_render()
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
				return self._default_render()

	def _health(self):
		response = send_get_request('health', self.request.POST)
		if response.ok():
			response = response.content()
			update_container(response['id'], {'last_check_at': timezone.now(), 'health_status': response['status']})
		else:
			self.message = response.status()
			return self._default_render()

	def _remove(self):
		self.message = stop_container(self.request.POST).status()
		return self._default_render()

	def _tensorboard_link(self):
		response = send_get_request('data/tensorboard', self.request.POST)
		if response.ok():
			return redirect(response.content()['data'])
		else:
			self.message = response.status()
			return self._default_render()
