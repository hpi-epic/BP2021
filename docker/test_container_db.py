import os
import sqlite3

from container_db_manager import ContainerDBManager

db_name = 'test_database.db'
test_database_name = os.path.join(os.path.dirname(__file__), db_name)
container_db_manager = None


def create_connection():
	db = None
	try:
		db = sqlite3.connect(test_database_name)
		cursor = db.cursor()
		return cursor, db
	except Exception as error:
		print(f'Could not connect to the database {error}')
		assert False


def tear_down_connection(db, cursor):
	try:
		cursor.close()
		db.close()
	except Exception as error:
		print(f'Could not disconnect from the database: {error}')
		assert False


def insert(given_id, container_id, status):
	cursor, db = create_connection()
	try:
		cursor.execute(
			"""
			INSERT INTO container VALUES (:given_id, :container_id, :status)
			""", {'given_id': given_id, 'container_id': container_id, 'status': status}
		)
		db.commit()
	except Exception as error:
		assert False, f'Could not insert: {error}'
	tear_down_connection(db, cursor)


def setup_function(function):
	global container_db_manager
	container_db_manager = ContainerDBManager(test_database_name)
	cursor, db = create_connection()
	cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=:name', {'name': 'container'})
	data = cursor.fetchone()
	assert data
	insert('test_id', '1234', 'sleeping')
	tear_down_connection(db, cursor)


def teardown_function(function):
	os.remove(test_database_name)


def test_get_translated_container_id():
	assert '1234' == container_db_manager.get_translated_container('test_id')


def test_get_translated_container_id_does_not_exist():
	assert container_db_manager.get_translated_container('abc') is None


def test_insert_n_container():
	result = container_db_manager.insert_n_container(3)
	assert all(given_id.startswith('recommerce') for given_id in result)
	assert 3 == len(result)


def test_update_container_id():
	insert('test_id2', '', '')
	assert container_db_manager.set_container_id('test_id2', 'abc')
	assert 'abc' == container_db_manager.get_translated_container('test_id2')
