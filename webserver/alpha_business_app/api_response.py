class APIResponse():
	def __init__(self,
			status: str,
			http_status: str = None,
			string_response: str = None,
			json_response: dict = None,
			raw_response=None) -> None:

		assert status == 'error' or status == 'success', 'as status only error and success are allowed'
		self.status_code = status
		self.http_status = http_status
		self.string_response = string_response if string_response else None
		self.json_response = json_response if json_response else None
		self.raw_response = raw_response if raw_response else None
		assert self._only_one_response_content_set(), 'you are only allowed to set one response content'

	def content(self):
		content_possibilities = [self.string_response, self.json_response, self.raw_response]
		for content in content_possibilities:
			if content:
				return content

	def ok(self) -> bool:
		print(self.status_code)
		if self.status_code == 'success':
			return True
		else:
			return False

	def not_found(self) -> bool:
		return self.http_status == 404

	def status(self) -> list:
		return [self.status_code, self.content()]

	def _only_one_response_content_set(self) -> bool:
		if self.string_response and self.json_response is None and self.raw_response is None:
			return True
		elif self.json_response and self.string_response is None and self.raw_response is None:
			return True
		elif self.raw_response and self.string_response is None and self.json_response is None:
			return True
		else:
			return False
