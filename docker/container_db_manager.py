# import datetime
import sqlite3
from datetime import datetime
from uuid import uuid4

from utils import bcolors


class ContainerDBRow:
	def __init__(self,
		_container_id: str = '',
		_config: str = '',
		_started_at: str = '',
		_started_by: str = '',
		_group_id: str = uuid4(),
		_group_member: int = 1,
		_stopped_at: str = '',
		_force_stop: str = '',
		_exited_at: str = '',
		_exit_status: str = '',
		_health: str = '',
		_paused: str = '',
		_resumed: str = '',
		_tensorboard: str = '',
		_logs: str = '',
		_data: str = '') -> None:

		self.container_id = str(_container_id)
		self.config = str(_config)
		self.started_at = str(_started_at)
		self.started_by = str(_started_by)
		self.group_id = str(_group_id)
		self.group_member = str(_group_member)
		self.stopped_at = str(_stopped_at)
		self.force_stop = str(_force_stop)
		self.exited_at = str(_exited_at)
		self.exit_status = str(_exit_status)
		self.health = str(_health)
		self.paused = str(_paused)
		self.resumed = str(_resumed)
		self.tensorboard = str(_tensorboard)
		self.logs = str(_logs)
		self.data = str(_data)

	def column_names_as_dict(self) -> dict:
		return {a: a for a in vars(self).keys()}

	def header_row(self) -> list:
		return list(vars(self).keys())

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
		return [list(row) for row in data]

	def get_csv_data(self):
		header = ContainerDBRow().header_row()
		data = self.get_all_container()
		final_data = ''
		for line in [header] + data:
			print(line)
			all_str = [str(item) for item in line]
			final_data += ';'.join(all_str) + '\n'
		return final_data.strip()

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

	def has_been_stopped(self, container_id: str, status_before_stop: str, status_after_stop: str, exit_status: str):
		print(status_before_stop)
		print(status_after_stop)
		print()
		self._update_value('stopped_at', datetime.now(), container_id, should_append=False)
		self._update_value('exit_status', exit_status, container_id, should_append=False)
		has_been_forced_stopped = status_before_stop != 'exited'
		self._update_value('force_stop', has_been_forced_stopped, container_id, should_append=False)

	def they_are_exited(self, exited_container: list):
		for container_id, status in exited_container:
			self._update_value('exited_at', datetime.now(), container_id, should_append=False)
			self._update_value('exit_status', status, container_id, should_append=False)
			self._update_value('force_stop', False, container_id, should_append=False)

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

	def _update_value(self, key_to_update: str, value_to_update, container_id: str, should_append: bool = True):
		# figure out if value already exists
		previous_value = self._select_value(key_to_update, container_id)
		if should_append:
			new_value = ';'.join([previous_value, str(value_to_update)]) if previous_value else value_to_update
		elif previous_value:
			# we are not supposed to append the new value and there is already an existing value
			return
		else:
			new_value = value_to_update
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
