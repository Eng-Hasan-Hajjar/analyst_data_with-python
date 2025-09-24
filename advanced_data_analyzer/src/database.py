import sqlite3
import json

class AdvancedDataAnalysisDB:
    def __init__(self):
        self.conn = sqlite3.connect('advanced_data_analysis.db', check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        tables = [
            '''CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_shape TEXT, analysis_type TEXT, parameters TEXT)''',
            
            '''CREATE TABLE IF NOT EXISTS saved_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT,
                model_type TEXT, accuracy REAL, parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                snapshot_name TEXT, data_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS user_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                note_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                action_type TEXT, action_details TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'''
        ]
        
        for table in tables:
            cursor.execute(table)
        self.conn.commit()
    
    def save_analysis_session(self, session_name: str, data_shape: tuple, analysis_type: str, parameters: dict) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO analysis_sessions (session_name, data_shape, analysis_type, parameters)
                         VALUES (?, ?, ?, ?)''', 
                      (session_name, str(data_shape), analysis_type, json.dumps(parameters)))
        self.conn.commit()
        return cursor.lastrowid