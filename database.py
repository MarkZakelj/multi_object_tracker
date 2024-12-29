import sqlite3
from threading import Lock
from datetime import datetime

DEFAULT_DB_PATH = 'tracking2.db'

class DBInterface:
    _instance = None
    _connection = None
    _lock = Lock()

    def __init__(cls, db_path):
        if cls._instance is None:
            cls._instance = super(DBInterface, cls).__new__(cls)
            cls._instance._db_path = db_path
            cls._instance._connect()
            cls._instance.db_create()
        return cls._instance

    def _connect(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self._db_path)

    def get_cursor(self):
        if self._connection is None:
            self._connect()
        return self._connection.cursor()

    def db_create(self):
        cursor = self.get_cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id TEXT,
                object_name TEXT,
                track_id INTEGER,
                confidence REAL,
                video_id TEXT,
                frame INTEGER,
                l REAL,
                t REAL, 
                r REAL,
                b REAL,
                timestamp DATETIME
            )
        ''')

        self._connection.commit()

    def add_detection(self, object_id, object_name, track_id, confidence, video_id, frame, bbox, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        cursor = self.get_cursor()
        
        cursor.execute('''
            INSERT INTO tracking 
            (object_id, object_name, track_id, confidence, video_id, frame, l, t, r, b, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (object_id, object_name, track_id, confidence, video_id, frame, 
              bbox[0], bbox[1], bbox[2], bbox[3], timestamp))
        self._connection.commit()

    def __del__(self):
        if self._connection:
            self._connection.close()
            self._connection = None

# THREAD SAFE functions (for streamlit)
         
class ThreadSafeDBInterface:
    _instance = None
    _lock = Lock()
    
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        
    def __enter__(self):
        self._lock.acquire()
        self.conn = sqlite3.connect(self.db_path)
        return self.conn.cursor()
        
    def __exit__(self, *args):
        self.conn.commit()
        self.conn.close()
        self._lock.release()
        
def get_videos():
    with ThreadSafeDBInterface() as cursor:
        cursor.execute("SELECT DISTINCT video_id FROM tracking")
        return [str(row[0]) for row in cursor.fetchall()]

def get_track_ids(video_id):
    with ThreadSafeDBInterface() as cursor:
        cursor.execute("SELECT DISTINCT track_id FROM tracking WHERE video_id = ?", (video_id,))
        return [str(row[0]) for row in cursor.fetchall()]

def get_trajectory_with_confidence(track_id):
    with ThreadSafeDBInterface() as cursor:
        cursor.execute("""
            SELECT frame, l, t, r, b, confidence
            FROM tracking 
            WHERE track_id = ? 
            ORDER BY frame ASC
        """, (track_id,))
        return cursor.fetchall()
