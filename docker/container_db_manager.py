import sqlite3
from uuid import uuid4

from docker_info import DockerInfo


class ContainerDBRow:
	def __init__(self, database_row: tuple or None) -> None:
		if not database_row:
			database_row = (None, None, None)
		self.given_id = database_row[0]
		self.container_id = database_row[1]
		self.status = database_row[2]

	@staticmethod
	def sql_schema() -> str:
		return '(given_id TEXT PRIMARY KEY, container_id TEXT, status TEXT)'

	def table_columns() -> str:
		return '(:given_id, :container_id, :status)'

	def __str__(self) -> str:
		return f'given: {self.given_id}, container_id: {self.container_id}'


class ContainerDBManager:
	def __init__(self, db_file_name: str = 'sqlite.db') -> None:
		self.db_file = db_file_name
		self.table_name = 'container'
		if not self._does_table_exist(self.table_name):
			self._create_table(self.table_name)

	def get_translated_container(self, given_id: str) -> str or None:
		"""
		Translates a given id into a real docker id.

		Args:
			given_id (str): id starting with recommerce that should be translated

		Returns:
			str or None: real container id or None if not exists
		"""
		data = self._select_value(given_id)
		return data.container_id

	def insert_n_container(self, num_container_to_be_inserted: int) -> list:
		result = []
		for _ in range(num_container_to_be_inserted):
			given_id = f'recommerce-{uuid4()}'
			self._insert_into_database(given_id, '')
			result += [given_id]
		return result

	def update_container_id(self, given_id: str, container_id: str) -> None:
		print('++', self._select_value(given_id))
		print('bevor update: given', given_id, 'c_id', container_id)
		self._update_value('container_id', container_id, given_id)
		print('++', self._select_value(given_id))

	def translate_n_container(self, container_info_dict: dict) -> dict:
		"""
		Translates a dictionary of DockerInfo dicts into a dictionary of DockerInfo dicts, but with given ids instead of real ids.

		Args:
			container_info_dict (dict): a dictionary of DockerInfo dicts

		Returns:
			list: a dictionary of DockerInfo dicts
		"""
		resulting_container_info = {}
		for index, docker_info in sorted(container_info_dict.items(), key=lambda tup: tup[0]):
			given_id = f'recommerce-{uuid4()}'
			self._insert_into_database(given_id, docker_info['id'], 'running')
			resulting_container_info[index] = vars(DockerInfo(given_id, docker_info['status'], docker_info['data'], docker_info['stream']))
		return resulting_container_info

	def set_container_id(self, given_id: str, new_container_id: str) -> bool:
		"""
		Sets container_id of a container to a new id for a given_id.

		Args:
			given_id (str): our custom id, should be in database
			new_container_id (str): value the container id should be set to

		Returns:
			bool: indecates if this was successfull.
		"""
		try:
			self._update_value('container_id', new_container_id, given_id)
			self._update_value('status', 'running', given_id)
		except Exception:
			return False
		return True

	def _create_connection(self) -> tuple:
		"""Connects to self.db_file.

		Returns:
			tuple: db and cursor if successfull, otherwise (None, None)
		"""
		db = None
		try:
			db = sqlite3.connect(self.db_file)
			cursor = db.cursor()
			return cursor, db
		except Exception as error:
			print(f'Could not connect to the database {error}')
		return None, None

	def _create_table(self, table_name: str) -> None:
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				CREATE TABLE {table_name} {ContainerDBRow.sql_schema()};
				"""
			)
			db.commit()
			print(f'created table {table_name} ')
		except Exception as error:
			print(f'Could not create database table. {error}')
		self._tear_down_connection(db, cursor)

	def _does_table_exist(self, table_name: str) -> bool:
		"""checks if table with `table_name` exists in database

		Args:
			table_name (str): name of the table that should be checked

		Returns:
			bool: indecates if table exists or not
		"""
		cursor, db = self._create_connection()
		cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=:name', {'name': table_name})
		data = cursor.fetchone()
		self._tear_down_connection(db, cursor)
		return bool(data)

	def _insert_into_database(self, given_id: str, container_id: str, status: str = 'waiting') -> bool:
		"""inserts a new row into database.

		Args:
			given_id (str): given_id of new row
			container_id (str): container_id of new row
			status (str): status of new row. Defaults to 'waiting'

		Returns:
			bool: indecates if value was correctly inserted
		"""
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				INSERT INTO {self.table_name} VALUES {ContainerDBRow.table_columns()}
				""", {'given_id': given_id, 'container_id': container_id, 'status': status}
			)
			db.commit()
		except Exception as error:
			print(f'Could not insert value into database: {error}')
			return False
		self._tear_down_connection(db, cursor)
		return True

	def _select_value(self, given_id: str) -> ContainerDBRow:
		"""selects database row by given_id.

		Args:
			given_id (str): given_id of new row

		Returns:
			ContainerDBRow: row as ContainerDBRow
		"""
		data = ''
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				f"""
				SELECT * FROM {self.table_name} WHERE given_id = :given_id
				""", {'given_id': given_id}
			)
			data = cursor.fetchone()
		except Exception as error:
			print(f'Could not select value: {error}')
		self._tear_down_connection(db, cursor)
		return ContainerDBRow(data)

	def _tear_down_connection(self, db, cursor):
		"""Stops connection to database.

		Args:
			db: database object
			cursor: database cursor
		"""
		try:
			cursor.close()
			db.close()
		except Exception as error:
			print(f'Could not disconnect from the database: {error}')

	def _update_value(self, key_to_update: str, value_to_update: str, given_id: str):
		cursor, db = self._create_connection()
		cursor.execute(
			f"""
			UPDATE {self.table_name} SET {key_to_update} = :value WHERE given_id = :given_id
			""", {'value': value_to_update, 'given_id': given_id}
		)
		db.commit()
		self._tear_down_connection(db, cursor)
