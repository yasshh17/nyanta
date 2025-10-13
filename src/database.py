"""Database module for chat history and document tracking."""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class ChatDatabase:
    """SQLite database for chat history and document tracking."""
    
    def __init__(self, db_path: str = "chat_data.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Chat messages table
        c.execute('''CREATE TABLE IF NOT EXISTS messages
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT NOT NULL,
                      role TEXT NOT NULL,
                      content TEXT NOT NULL,
                      sources TEXT,
                      timestamp TEXT NOT NULL)''')
        
        # Documents table
        c.execute('''CREATE TABLE IF NOT EXISTS documents
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      filename TEXT NOT NULL,
                      file_size INTEGER,
                      chunk_count INTEGER,
                      upload_timestamp TEXT NOT NULL,
                      status TEXT DEFAULT 'active')''')
        
        # Sessions table
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (session_id TEXT PRIMARY KEY,
                      created_at TEXT NOT NULL,
                      last_activity TEXT NOT NULL,
                      message_count INTEGER DEFAULT 0)''')
        
        conn.commit()
        conn.close()
    
    def save_message(self, session_id: str, role: str, content: str, 
                    sources: Optional[List[Dict]] = None):
        """Save a chat message."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""INSERT INTO messages 
                     (session_id, role, content, sources, timestamp) 
                     VALUES (?, ?, ?, ?, ?)""",
                  (session_id, role, content, 
                   json.dumps(sources) if sources else None,
                   datetime.now().isoformat()))
        
        # Update session
        c.execute("""INSERT OR REPLACE INTO sessions 
                     (session_id, created_at, last_activity, message_count)
                     VALUES (?, 
                             COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), ?),
                             ?,
                             COALESCE((SELECT message_count FROM sessions WHERE session_id = ?), 0) + 1)""",
                  (session_id, session_id, datetime.now().isoformat(), 
                   datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
    
    def load_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Load chat messages for a session."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT role, content, sources, timestamp 
                     FROM messages 
                     WHERE session_id = ? 
                     ORDER BY id DESC LIMIT ?""", (session_id, limit))
        
        messages = []
        for row in reversed(c.fetchall()):
            msg = {
                "role": row[0],
                "content": row[1],
                "timestamp": row[3]
            }
            if row[2]:
                msg["sources"] = json.loads(row[2])
            messages.append(msg)
        
        conn.close()
        return messages
    
    def save_document(self, filename: str, file_size: int, chunk_count: int):
        """Save document metadata."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""INSERT INTO documents 
                     (filename, file_size, chunk_count, upload_timestamp) 
                     VALUES (?, ?, ?, ?)""",
                  (filename, file_size, chunk_count, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_document_stats(self) -> Dict:
        """Get document statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*), SUM(chunk_count) FROM documents WHERE status='active'")
        result = c.fetchone()
        
        conn.close()
        return {
            "total_documents": result[0] or 0,
            "total_chunks": result[1] or 0
        }
    
    def clear_session(self, session_id: str):
        """Clear messages for a session."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all chat sessions."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT session_id, created_at, last_activity, message_count 
                     FROM sessions 
                     ORDER BY last_activity DESC""")
        
        sessions = []
        for row in c.fetchall():
            sessions.append({
                "session_id": row[0],
                "created_at": row[1],
                "last_activity": row[2],
                "message_count": row[3]
            })
        
        conn.close()
        return sessions
    
    def get_most_recent_session(self) -> Optional[str]:
        """Get the most recently active session ID."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT session_id FROM sessions 
                     ORDER BY last_activity DESC LIMIT 1""")
        
        result = c.fetchone()
        conn.close()
        
        return result[0] if result else None