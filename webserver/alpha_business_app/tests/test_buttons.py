from django.test import TestCase
from django.core.handlers.wsgi import WSGIRequest
from django.test import RequestFactory, Client
from ..models import Container, update_container

# from ..buttons import ButtonHandler


class ButtonsTest(TestCase):

	def setUp(self):
		Container.objects.create(
            command='training',
            container_id='123',
            created_at='01.01.1970',
            last_check_at='now',
            name='test_container'
        )
# 	def __init__(self, request, view: str, container: Container = None, rendering_method: str = 'default', data: str = None) -> None:

	def test_health(self):
		# request = WSGIRequest.object.get(POST={'csrfmiddlewaretoken': ['EpxeVzX46YD1xP61FqYY9Ty7NRB6vgx4igpHTB3ns6xuN7iAn8AGWu594tZPRLCp'], 'action': ['health'], 'container_id': ['0237ebb7847353fd703008a087b1654f4f56563d6931c3506ae058bba9eb0027']})
		# test_button_handler: ButtonHandler = ButtonHandler.object.get(request="<POST '/observe'>", view='observe.html')
		# test_button_handler.handle_button_click()
		c = Client()
		response = c.post('/observe',{'action': ['health'], 'container_id': ['123']})
		print("response status code:", response.status_code)
		assert response.status_code == 200

	def test_pause(self):
		c = Client()
		response = c.post('/observe', {'action': ['unpause'], 'container_id': ['123']})
		print("response status code:", response.status_code)
		assert response.status_code == 200

if __name__ == '__main__':
	bt = ButtonsTest()
	bt.test_health()
