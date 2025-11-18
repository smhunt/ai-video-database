"""Database models for video chat system using SQLite."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger
import json

DATABASE_PATH = "data/video_chat.db"


class VideoDatabase:
    """SQLite database manager for video chat system."""

    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.connection.cursor()

        # Videos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                duration_seconds REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                size_bytes INTEGER,
                path TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processing',
                metadata TEXT
            )
        """)

        # Frames table - stores extracted frames with analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                timestamp_seconds REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                frame_path TEXT NOT NULL,
                description TEXT,
                objects TEXT,
                actions TEXT,
                scene_type TEXT,
                is_keyframe BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        """)

        # Frame embeddings table - for semantic search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                embedding_vector TEXT NOT NULL,
                model_name TEXT DEFAULT 'mxbai-embed-large-v1',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (frame_id) REFERENCES frames (id) ON DELETE CASCADE
            )
        """)

        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                relevant_frames TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        """)

        # Cost tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                operation_type TEXT NOT NULL,
                api_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                num_images INTEGER DEFAULT 0,
                num_embeddings INTEGER DEFAULT 0,
                estimated_cost_usd REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        """)

        # Transcripts table - stores audio transcriptions with timestamps
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                text TEXT NOT NULL,
                language TEXT,
                confidence REAL,
                model_name TEXT DEFAULT 'whisper-1',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frames_video_id
            ON frames(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frames_timestamp
            ON frames(timestamp_seconds)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_frame_id
            ON frame_embeddings(frame_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_video_id
            ON chat_history(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_costs_video_id
            ON api_costs(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_costs_timestamp
            ON api_costs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcripts_video_id
            ON transcripts(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcripts_time
            ON transcripts(start_time, end_time)
        """)

        self.connection.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def add_video(
        self,
        filename: str,
        original_filename: str,
        path: str,
        size_bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new video record and return its ID."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO videos (filename, original_filename, path, size_bytes, metadata, status)
            VALUES (?, ?, ?, ?, ?, 'processing')
        """,
            (filename, original_filename, path, size_bytes, json.dumps(metadata or {})),
        )
        self.connection.commit()
        video_id = cursor.lastrowid
        logger.info(f"Added video {original_filename} with ID {video_id}")
        return video_id

    def update_video_info(
        self,
        video_id: int,
        duration_seconds: float,
        fps: float,
        width: int,
        height: int,
    ):
        """Update video metadata after processing."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE videos
            SET duration_seconds = ?, fps = ?, width = ?, height = ?
            WHERE id = ?
        """,
            (duration_seconds, fps, width, height, video_id),
        )
        self.connection.commit()
        logger.info(f"Updated video {video_id} metadata")

    def update_video_status(self, video_id: int, status: str):
        """Update video processing status."""
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE videos SET status = ? WHERE id = ?",
            (status, video_id),
        )
        self.connection.commit()
        logger.info(f"Updated video {video_id} status to {status}")

    def add_frame(
        self,
        video_id: int,
        timestamp_seconds: float,
        frame_number: int,
        frame_path: str,
        description: Optional[str] = None,
        objects: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        scene_type: Optional[str] = None,
        is_keyframe: bool = False,
    ) -> int:
        """Add a frame record and return its ID."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO frames (
                video_id, timestamp_seconds, frame_number, frame_path,
                description, objects, actions, scene_type, is_keyframe
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                video_id,
                timestamp_seconds,
                frame_number,
                frame_path,
                description,
                json.dumps(objects or []),
                json.dumps(actions or []),
                scene_type,
                is_keyframe,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def add_frame_embedding(self, frame_id: int, embedding: List[float], model_name: str):
        """Add embedding vector for a frame."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO frame_embeddings (frame_id, embedding_vector, model_name)
            VALUES (?, ?, ?)
        """,
            (frame_id, json.dumps(embedding), model_name),
        )
        self.connection.commit()

    def get_video(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get video by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_all_videos(self) -> List[Dict[str, Any]]:
        """Get all videos."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY upload_date DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_frames_for_video(self, video_id: int) -> List[Dict[str, Any]]:
        """Get all frames for a video."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM frames
            WHERE video_id = ?
            ORDER BY timestamp_seconds ASC
        """,
            (video_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_keyframes_for_video(self, video_id: int) -> List[Dict[str, Any]]:
        """Get only keyframes for a video."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM frames
            WHERE video_id = ? AND is_keyframe = 1
            ORDER BY timestamp_seconds ASC
        """,
            (video_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_frame_at_timestamp(
        self, video_id: int, timestamp: float, tolerance: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Get frame closest to given timestamp within tolerance."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM frames
            WHERE video_id = ?
            AND ABS(timestamp_seconds - ?) <= ?
            ORDER BY ABS(timestamp_seconds - ?) ASC
            LIMIT 1
        """,
            (video_id, timestamp, tolerance, timestamp),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_frame_embedding(self, frame_id: int) -> Optional[List[float]]:
        """Get embedding vector for a frame."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT embedding_vector FROM frame_embeddings WHERE frame_id = ?",
            (frame_id,),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def add_chat_message(
        self, video_id: int, query: str, response: str, relevant_frames: Optional[List[int]] = None
    ):
        """Add a chat message to history."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (video_id, query, response, relevant_frames)
            VALUES (?, ?, ?, ?)
        """,
            (video_id, query, response, json.dumps(relevant_frames or [])),
        )
        self.connection.commit()

    def get_chat_history(self, video_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a video."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM chat_history
            WHERE video_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (video_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()][::-1]  # Reverse to chronological

    def add_api_cost(
        self,
        operation_type: str,
        api_provider: str,
        model_name: str,
        estimated_cost_usd: float,
        video_id: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_images: int = 0,
        num_embeddings: int = 0,
        details: Optional[str] = None,
    ):
        """Add an API cost record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO api_costs (
                video_id, operation_type, api_provider, model_name,
                input_tokens, output_tokens, num_images, num_embeddings,
                estimated_cost_usd, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                video_id,
                operation_type,
                api_provider,
                model_name,
                input_tokens,
                output_tokens,
                num_images,
                num_embeddings,
                estimated_cost_usd,
                details,
            ),
        )
        self.connection.commit()

    def get_costs_for_video(self, video_id: int) -> List[Dict[str, Any]]:
        """Get all API costs for a video."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM api_costs
            WHERE video_id = ?
            ORDER BY timestamp ASC
        """,
            (video_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_total_costs(self) -> Dict[str, Any]:
        """Get total costs across all operations."""
        cursor = self.connection.cursor()

        # Total cost
        cursor.execute("SELECT SUM(estimated_cost_usd) FROM api_costs")
        total_cost = cursor.fetchone()[0] or 0.0

        # Cost by provider
        cursor.execute("""
            SELECT api_provider, SUM(estimated_cost_usd) as cost
            FROM api_costs
            GROUP BY api_provider
        """)
        by_provider = {row[0]: row[1] for row in cursor.fetchall()}

        # Cost by operation
        cursor.execute("""
            SELECT operation_type, SUM(estimated_cost_usd) as cost
            FROM api_costs
            GROUP BY operation_type
        """)
        by_operation = {row[0]: row[1] for row in cursor.fetchall()}

        # Recent costs (last 24 hours)
        cursor.execute("""
            SELECT SUM(estimated_cost_usd)
            FROM api_costs
            WHERE timestamp >= datetime('now', '-1 day')
        """)
        last_24h = cursor.fetchone()[0] or 0.0

        return {
            "total_cost_usd": round(total_cost, 4),
            "last_24h_usd": round(last_24h, 4),
            "by_provider": by_provider,
            "by_operation": by_operation,
        }

    def delete_video(self, video_id: int):
        """Delete video and all associated data."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        self.connection.commit()
        logger.info(f"Deleted video {video_id}")

    def add_transcript_segment(
        self,
        video_id: int,
        start_time: float,
        end_time: float,
        text: str,
        language: Optional[str] = None,
        confidence: Optional[float] = None,
        model_name: str = "whisper-1",
    ) -> int:
        """Add a transcript segment and return its ID."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO transcripts (
                video_id, start_time, end_time, text, language, confidence, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (video_id, start_time, end_time, text, language, confidence, model_name),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_transcripts_for_video(self, video_id: int) -> List[Dict[str, Any]]:
        """Get all transcript segments for a video."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM transcripts
            WHERE video_id = ?
            ORDER BY start_time ASC
        """,
            (video_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_transcript_at_timestamp(
        self, video_id: int, timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """Get transcript segment at specific timestamp."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM transcripts
            WHERE video_id = ?
            AND start_time <= ?
            AND end_time >= ?
            LIMIT 1
        """,
            (video_id, timestamp, timestamp),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_full_transcript_text(self, video_id: int) -> str:
        """Get full transcript as single text string."""
        segments = self.get_transcripts_for_video(video_id)
        return " ".join(seg["text"] for seg in segments)

    def search_transcript(
        self, video_id: int, search_term: str
    ) -> List[Dict[str, Any]]:
        """Search for text in transcript segments."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM transcripts
            WHERE video_id = ?
            AND text LIKE ?
            ORDER BY start_time ASC
        """,
            (video_id, f"%{search_term}%"),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global database instance
_db_instance: Optional[VideoDatabase] = None


def get_db() -> VideoDatabase:
    """Get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = VideoDatabase()
    return _db_instance
