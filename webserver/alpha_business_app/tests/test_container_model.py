from django.test import TestCase
from django.utils import timezone

from ..models import Container, update_container


class ContainerTest(TestCase):
	def setUp(self):
		Container.objects.create(
								command='training',
								container_id='123',
								created_at='01.01.1970',
								last_check_at='now',
								name='test_container'
								)

	def test_container_id(self):
		test_container: Container = Container.objects.get(container_id='123')
		assert '123' == test_container.id()

	def test_container__is_archived(self):
		test_container: Container = Container.objects.get(container_id='123')
		assert test_container.is_archived() is False

		test_container.health_status = 'archived'
		assert test_container.is_archived() is True

	def test_tensorboard_link(self):
		test_container: Container = Container.objects.get(container_id='123')
		assert test_container.has_tensorboard_link() is False

		test_container.tensorboard_link = 'http://test-example.com'

		assert test_container.has_tensorboard_link() is True

	def test_update_container(self):
		check_time = timezone.now()
		update_dict = {'health_status': 'test',
						'last_check_at': check_time}
		update_container('123', update_dict)

		test_container: Container = Container.objects.get(container_id='123')
		assert check_time == test_container.last_check_at
		assert 'test' == test_container.health_status

		update_dict = {'tensorboard_link': 'test'}
		update_container('123', update_dict)

		test_container: Container = Container.objects.get(container_id='123')
		assert test_container.has_tensorboard_link() is True
		assert 'test' == test_container.tensorboard_link
