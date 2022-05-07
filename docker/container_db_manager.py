# import datetime
import sqlite3
from datetime import datetime
from uuid import uuid4

from utils import bcolors


class ContainerDBRow:
	def __init__(self,
		_container_id: str = None,
		_config: str = None,
		_started_at: str = None,
		_started_by: str = None,
		_group_id: str = uuid4(),
		_group_member: int = 1,
		_stopped_at: str = None,
		_force_stop: str = None,
		_health: str = None,
		_paused: str = None,
		_resumed: str = None,
		_tensorboard: str = None,
		_logs: str = None,
		_data: str = None) -> None:

		self.container_id = str(_container_id) if _container_id else ''
		self.config = str(_config) if _config else ''
		self.started_at = str(_started_at) if _started_at else ''
		self.started_by = str(_started_by) if _started_by else ''
		self.group_id = str(_group_id)
		self.group_member = str(_group_member)
		self.stopped_at = str(_stopped_at) if _stopped_at else ''
		self.force_stop = str(_force_stop) if _force_stop else ''
		self.health = str(_health) if _health else ''
		self.paused = str(_paused) if _paused else ''
		self.resumed = str(_resumed) if _resumed else ''
		self.tensorboard = str(_tensorboard) if _tensorboard else ''
		self.logs = str(_logs) if _logs else ''
		self.data = str(_data) if _data else ''

	def column_names(self) -> tuple:
		return {a: a for a in vars(self).keys()}

	def sql_column_names(self) -> str:
		return str(tuple([':' + k for k in vars(self).keys()])).replace("'", '')

	def schema(self) -> str:
		all_keys = list(vars(self).keys())
		all_keys[0] += ' TEXT PRIMARY KEY'
		return ', '.join(all_keys)


class ContainerDB:
	def __init__(self) -> None:
		self.db_file = 'sqlite.db'
		self.table_name = 'container'
		if not self._does_table_exist():
			self._create_container_table()

	def get_all_container(self):
		data = None
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				"""
				SELECT * FROM container
				"""
			)
			data = cursor.fetchall()
		except Exception as error:
			print(f'{bcolors.FAIL}Could not insert value into database: {error}{bcolors.ENDC}')

		self._tear_down_connection(db, cursor)
		return data

	def insert(self, all_container_infos, starting_time, is_webserver_user: bool, config: dict) -> None:
		container_starter = 'websrv.eaalab' if is_webserver_user else 'dev'
		group_size = len(all_container_infos)
		group_id = uuid4()
		for container_info in all_container_infos:
			current_container = ContainerDBRow(container_info.id, config, starting_time, container_starter, group_id, group_size)
			self._insert_into_database(current_container)

	def has_been_paused(self, container_id: str):
		self._update_value('paused', datetime.now(), container_id)

	def has_got_tensorboard(self, container_id: str):
		self._update_value('tensorboard', datetime.now(), container_id)

	def has_got_data(self, container_id: str):
		self._update_value('data', datetime.now(), container_id)

	def has_been_health_checked(self, container_id: str):
		self._update_value('health', datetime.now(), container_id)

	def has_been_unpaused(self, container_id: str):
		self._update_value('resumed', datetime.now(), container_id)

	def has_got_logs(self, container_id: str):
		self._update_value('logs', datetime.now(), container_id)

	def has_been_stopped(self, container_id: str, status_before_checked: str):
		self._update_value('stopped_at', datetime.now(), container_id)
		has_been_forced_stopped = status_before_checked != 'exited'
		self._update_value('force_stop', has_been_forced_stopped, container_id)

	def _create_connection(self):
		db = None
		try:
			db = sqlite3.connect(self.db_file)
			cursor = db.cursor()
			return cursor, db
		except Exception as error:
			print(f'{bcolors.FAIL}Could not connect to the database {error}{bcolors.ENDC}')
		return None, None

	def _create_container_table(self):
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				CREATE TABLE {self.table_name}
					({ContainerDBRow().schema()})
				"""
			)
			db.commit()
			print(f'{bcolors.OKGREEN} created table {self.table_name} {bcolors.ENDC}')
		except Exception as error:
			print(f'{bcolors.FAIL}Could not create database table. {error}{bcolors.ENDC}')
		self._tear_down_connection(db, cursor)

	def _does_table_exist(self) -> bool:
		cursor, db = self._create_connection()
		cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=:name', {'name': self.table_name})
		data = cursor.fetchone()
		self._tear_down_connection(db, cursor)
		return bool(data)

	def _insert_into_database(self, container_row: ContainerDBRow):
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				INSERT INTO container VALUES {ContainerDBRow().sql_column_names()}
				""", vars(container_row)
			)
			db.commit()
		except Exception as error:
			print(f'{bcolors.FAIL}Could not insert value into database: {error}{bcolors.ENDC}')
		self._tear_down_connection(db, cursor)

	def _select_value(self, key_to_select: str, container_id: str) -> str:
		data = ''
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				SELECT {key_to_select} FROM {self.table_name} WHERE container_id = :container_id OR container_id=1234
				""", {'container_id': container_id}
			)
			data = cursor.fetchone()[0]
		except Exception as error:
			print(f'{bcolors.FAIL}Could not select value: {error}{bcolors.ENDC}')
		self._tear_down_connection(db, cursor)
		return data

	def _tear_down_connection(self, db, cursor):
		try:
			cursor.close()
			db.close()
		except Exception as error:
			print(f'{bcolors.FAIL}Could not disconnect from the database: {error}{bcolors.ENDC}')

	def _update_value(self, key_to_update: str, value_to_update, container_id: str):
		# figure out if value already exists
		previous_value = self._select_value(key_to_update, container_id)
		new_value = ';'.join([previous_value, str(value_to_update)]) if previous_value else value_to_update
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				UPDATE container SET {key_to_update} = :value WHERE container_id = :container_id
				""", {'value': new_value, 'container_id': container_id}
			)
			db.commit()
		except Exception as error:
			print(f'{bcolors.FAIL}Could not update value: {error}{bcolors.ENDC}')
		self._tear_down_connection(db, cursor)
