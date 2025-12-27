"""
vector_database.py - PostgreSQL Database Operations Module

This module handles all database operations for storing and retrieving
document chunks and their embeddings.

Usage:
    from vector_database import VectorDatabase
    from config import Config

    config = Config()
    with VectorDatabase(config) as db:
        db.save_chunk(text, embedding, filename, strategy)
"""

import json
import logging
from typing import List, Optional, Tuple

import psycopg2
from psycopg2 import sql

from config import Config


class VectorDatabase:
    """
    PostgreSQL database handler for document chunks and embeddings.

    This class is responsible for:
    - Managing database connections
    - Saving individual chunks with embeddings
    - Batch saving multiple chunks
    - Querying chunk counts and data
    - Using context manager pattern for connection cleanup

    The embeddings are stored as JSONB for compatibility without
    requiring the pgvector extension.

    Usage with context manager (recommended):
        with VectorDatabase(config) as db:
            db.save_chunk(text, embedding, filename, strategy)

    Usage without context manager:
        db = VectorDatabase(config)
        db.connect()
        db.save_chunk(text, embedding, filename, strategy)
        db.close()
    """

    def __init__(self, config: Config):
        """
        Initialize database with configuration.

        Args:
            config: Config object containing database credentials
        """
        self.config = config
        self.connection = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Context manager entry: establish database connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close database connection."""
        self.close()

    def connect(self):
        """
        Establish connection to PostgreSQL database.

        Raises:
            psycopg2.OperationalError: If connection fails
        """
        try:
            self.connection = psycopg2.connect(
                host=self.config.pg_host,
                port=self.config.pg_port,
                database=self.config.pg_database,
                user=self.config.pg_user,
                password=self.config.pg_password
            )
            self.logger.debug(
                f"Connected to database {self.config.pg_database} "
                f"at {self.config.pg_host}:{self.config.pg_port}"
            )
        except psycopg2.OperationalError as e:
            self.logger.error(f"Database connection failed: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.debug("Database connection closed")

    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self.connection is not None and not self.connection.closed

    def save_chunk(self, chunk_text: str, embedding: List[float],
                   filename: str, strategy: str) -> int:
        """
        Save a single chunk with its embedding to the database.

        Args:
            chunk_text: The text content of the chunk
            embedding: The embedding vector as a list of floats
            filename: Original document filename
            strategy: Chunking strategy used ('fixed', 'sentence', 'paragraph')

        Returns:
            The ID of the inserted row

        Raises:
            Exception: If insert fails
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected. Call connect() first.")

        cursor = self.connection.cursor()

        try:
            # Convert embedding list to JSON for JSONB storage
            embedding_json = json.dumps(embedding)

            cursor.execute("""
                INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (chunk_text, embedding_json, filename, strategy))

            row_id = cursor.fetchone()[0]
            self.connection.commit()

            self.logger.debug(f"Saved chunk {row_id} for file {filename}")
            return row_id

        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Failed to save chunk: {e}")
            raise
        finally:
            cursor.close()

    def save_chunks_batch(self, chunks_data: List[Tuple[str, List[float], str, str]]) -> int:
        """
        Save multiple chunks efficiently using batch insert.

        Args:
            chunks_data: List of tuples (chunk_text, embedding, filename, strategy)

        Returns:
            Number of rows inserted

        Raises:
            Exception: If batch insert fails
        """
        if not chunks_data:
            return 0

        if not self.is_connected():
            raise RuntimeError("Database not connected. Call connect() first.")

        cursor = self.connection.cursor()

        try:
            # Prepare data with JSON-encoded embeddings
            prepared_data = [
                (text, json.dumps(emb), fname, strat)
                for text, emb, fname, strat in chunks_data
            ]

            # Use executemany for batch insert
            cursor.executemany("""
                INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
                VALUES (%s, %s, %s, %s)
            """, prepared_data)

            row_count = cursor.rowcount
            self.connection.commit()

            self.logger.info(f"Batch saved {row_count} chunks")
            return row_count

        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Batch insert failed: {e}")
            raise
        finally:
            cursor.close()

    def get_chunk_count(self, filename: Optional[str] = None) -> int:
        """
        Get the count of chunks in the database.

        Args:
            filename: If provided, count only chunks from this file

        Returns:
            Number of chunks
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected. Call connect() first.")

        cursor = self.connection.cursor()

        try:
            if filename:
                cursor.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE filename = %s",
                    (filename,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM document_chunks")

            count = cursor.fetchone()[0]
            return count

        finally:
            cursor.close()

    def get_chunks_by_file(self, filename: str) -> List[dict]:
        """
        Get all chunks for a specific file.

        Args:
            filename: The filename to query

        Returns:
            List of dictionaries with chunk data
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected. Call connect() first.")

        cursor = self.connection.cursor()

        try:
            cursor.execute("""
                SELECT id, chunk_text, filename, split_strategy, created_at
                FROM document_chunks
                WHERE filename = %s
                ORDER BY id
            """, (filename,))

            rows = cursor.fetchall()
            chunks = []

            for row in rows:
                chunks.append({
                    'id': row[0],
                    'chunk_text': row[1],
                    'filename': row[2],
                    'split_strategy': row[3],
                    'created_at': row[4]
                })

            return chunks

        finally:
            cursor.close()

    def delete_chunks_by_file(self, filename: str) -> int:
        """
        Delete all chunks for a specific file.

        Useful for re-indexing a document.

        Args:
            filename: The filename whose chunks should be deleted

        Returns:
            Number of rows deleted
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected. Call connect() first.")

        cursor = self.connection.cursor()

        try:
            cursor.execute(
                "DELETE FROM document_chunks WHERE filename = %s",
                (filename,)
            )

            deleted_count = cursor.rowcount
            self.connection.commit()

            self.logger.info(f"Deleted {deleted_count} chunks for file {filename}")
            return deleted_count

        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Failed to delete chunks: {e}")
            raise
        finally:
            cursor.close()

    def get_all_filenames(self) -> List[str]:
        """
        Get list of all unique filenames in the database.

        Returns:
            List of filenames
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected. Call connect() first.")

        cursor = self.connection.cursor()

        try:
            cursor.execute(
                "SELECT DISTINCT filename FROM document_chunks ORDER BY filename"
            )

            filenames = [row[0] for row in cursor.fetchall()]
            return filenames

        finally:
            cursor.close()

    def test_connection(self) -> bool:
        """
        Test the database connection.

        Returns:
            True if connection is working, False otherwise
        """
        try:
            if not self.is_connected():
                self.connect()

            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

            return True

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
