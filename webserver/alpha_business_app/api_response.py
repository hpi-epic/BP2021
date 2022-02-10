class APIResponse():
	"""
	This class encapsulates responses from the docker API.
	"""
	def __init__(self, status: str,
			http_status: str = None, content=None) -> None:

		assert status == 'error' or status == 'success', 'as status only error and success are allowed'
		self.status_code = status
		self.http_status = http_status
		self.content = content
		assert self._only_one_response_content_set(), 'you are only allowed to set one response content'

	def ok(self) -> bool:
		return self.status_code == 'success'

	def not_found(self) -> bool:
		return self.http_status == 404

	def status(self) -> list:
		return [self.status_code, self.content]
