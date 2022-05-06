# import datetime
import sqlite3


class ContainerDBRow:
	def __init__(self,
		_container_id: str = None,
		_config: str = None,
		_started_at: str = None,
		_started_by: str = None,
		_stopped_at: str = None,
		_force_stop: str = None,
		_health: str = None,
		_paused: str = None,
		_resumed: str = None,
		_tensorboard: str = None,
		_logs: str = None,
		_data: str = None) -> None:

		self.container_id = _container_id if _container_id else ''
		self.config = _config if _config else ''
		self.started_at = _started_at if _started_at else ''
		self.started_by = _started_by if _started_by else ''
		self.stopped_at = _stopped_at if _stopped_at else ''
		self.force_stop = _force_stop if _force_stop else ''
		self.health = _health if _health else ''
		self.paused = _paused if _paused else ''
		self.resumed = _resumed if _resumed else ''
		self.tensorboard = _tensorboard if _tensorboard else ''
		self.logs = _logs if _logs else ''
		self.data = _data if _data else ''

	def to_tuple(self) -> tuple:
		return tuple(vars(self).values())

	@classmethod
	def column_names(cls) -> tuple:
		return tuple(vars(cls).keys())


class ContainerDB:
	def __init__(self) -> None:
		self.db_file = 'sqlite.db'
		self.table_name = 'container'
		cursor, db = self._create_connection()
		cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="?"', tuple(self.table_name))
		self._tear_down_connection(db, cursor)
		# self.db = sqlite3.connect(self.db_file)

	def get_all_container(self):
		pass

	def insert(self, all_container_infos, current_time, is_webserver_user: bool, config: dict) -> None:
		print(all_container_infos)
		cursor, db = self._create_connection()
		for container_info in all_container_infos:
			pass
		# self.container_table.insert({'bla': str(a)})
		self._tear_down_connection(db, cursor)

	def has_been_paused(self, container_id):
		pass

	def has_got_tensorboard(self, container_id):
		pass

	def has_got_data(self, container_id):
		pass

	def has_been_health_checked(self, container_id):
		pass

	def has_been_unpaused(self, container_id):
		pass

	def has_got_logs(self, container_id):
		pass

	def has_been_stopped(self, container_id, status_before_checked):
		pass

	def _create_connection(self):
		db = None
		try:
			db = sqlite3.connect(self.db_file)
			cursor = db.cursor()
			return cursor, db
		except Exception as e:
			print(f'Could not connect to the database {e}')
		return None, None

	def _create_container_table(self, db, cursor):
		try:
			cursor.execute(
				"""
				CREATE TABLE ? (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""", tuple(self.table_name) + ContainerDBRow.column_names())
		except Exception:
			print('Could not create database table.')

	def _insert_into_database(self, container_row: ContainerDBRow):
		cursor, db = self._create_connection()
		try:
			cursor.execute(
				"""
				INSERT INTO ? VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""", tuple(self.table_name) + container_row.to_tuple())
		except Exception:
			print('Could not insert value into database')
		self._tear_down_connection(db, cursor)

	def _tear_down_connection(self, db, cursor):
		try:
			cursor.close()
			db.close()
		except Exception:
			print('Could not disconnect from the database.')
