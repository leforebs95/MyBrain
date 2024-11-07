from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import text
from google.cloud.sql.connector import Connector
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from errors import DatabaseError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    instance_connection_name: str  # e.g., "project:region:instance"
    database: str
    user: str
    password: str
    pool_size: int = 5
    max_overflow: int = 2


class DatabaseManager:
    """
    Manages PostgreSQL database operations with support for vector storage.
    Uses Cloud SQL Connector and SQLAlchemy connection pooling.
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize database connection pool."""
        try:
            self.config = config
            self.connector = Connector()

            # Initialize the connection pool
            self.engine = sqlalchemy.create_engine(
                "postgresql+pg8000://",
                creator=self._get_connection,
                poolclass=QueuePool,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
            )

            # Test connection and initialize database
            with self.get_connection() as conn:
                self._init_database(conn)

            logger.info("Initialized DatabaseManager with connection pool")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _get_connection(self):
        """Create a new database connection using Cloud SQL Connector."""
        try:
            conn = self.connector.connect(
                self.config.instance_connection_name,
                "pg8000",
                user=self.config.user,
                password=self.config.password,
                db=self.config.database,
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {str(e)}")
            raise DatabaseError(f"Connection creation failed: {str(e)}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        try:
            with self.engine.connect() as connection:
                yield connection
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise DatabaseError(f"Database connection error: {str(e)}")

    def _init_database(self, connection):
        """Initialize database schema and extensions."""
        try:
            # Enable vector extension
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            # Create documents table
            connection.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    input_pdf TEXT NOT NULL,
                    output_base TEXT NOT NULL,
                    original_ocr TEXT,
                    improved_ocr TEXT,
                    embedding_vector vector(1024),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """
                )
            )

            # Create updated_at trigger
            connection.execute(
                text(
                    """
                CREATE OR REPLACE FUNCTION update_timestamp()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """
                )
            )

            connection.execute(
                text(
                    """
                DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW
                    EXECUTE PROCEDURE update_timestamp();
            """
                )
            )

            connection.commit()
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _format_vector(self, embedding: List[float]) -> str:
        """Format a vector for pgvector storage."""
        if embedding is None:
            return None
        return f"[{','.join(str(x) for x in embedding)}]"

    @retry_with_backoff()
    def store_document(self, doc_data: Dict) -> str:
        """Store document data in PostgreSQL."""
        try:
            logger.info(f"Storing document with data: {doc_data.keys()}")
            with self.get_connection() as conn:
                query = text(
                    """
                    INSERT INTO documents (
                    id, input_pdf, output_base, original_ocr, 
                    improved_ocr, embedding_vector, metadata
                ) VALUES (
                    :id, :input_pdf, :output_base, :original_ocr, 
                    :improved_ocr, :embedding_vector, :metadata
                )
                ON CONFLICT (id) DO UPDATE SET
                    input_pdf = EXCLUDED.input_pdf,
                    output_base = EXCLUDED.output_base,
                    original_ocr = EXCLUDED.original_ocr,
                    improved_ocr = EXCLUDED.improved_ocr,
                    embedding_vector = EXCLUDED.embedding_vector
                RETURNING id;
                """
                )

                result = conn.execute(
                    query,
                    {
                        "id": doc_data["id"],
                        "input_pdf": doc_data["input_pdf"],
                        "output_base": doc_data["output_base"],
                        "original_ocr": doc_data.get("original_ocr"),
                        "improved_ocr": doc_data.get("improved_ocr"),
                        "embedding_vector": self._format_vector(
                            doc_data.get("embedding")
                        ),
                        "metadata": doc_data.get("metadata", {}),
                    },
                )

                conn.commit()
                return result.scalar_one()

        except SQLAlchemyError as e:
            logger.error(f"Failed to store document: {str(e)}")
            raise DatabaseError(f"Failed to store document: {str(e)}")

    @retry_with_backoff()
    def find_similar_documents(
        self, embedding: List[float], limit: int = 10
    ) -> List[Dict]:
        """Find similar documents using vector similarity search."""
        try:
            with self.get_connection() as conn:
                query = text(
                    """
                    SELECT 
                        id, 
                        input_pdf,
                        improved_ocr,
                        metadata,
                        1 - (embedding_vector <=> :embedding) as similarity
                    FROM documents
                    WHERE embedding_vector IS NOT NULL
                    ORDER BY embedding_vector <=> :embedding
                    LIMIT :limit;
                """
                )

                result = conn.execute(query, {"embedding": embedding, "limit": limit})

                return [
                    {
                        "id": row.id,
                        "input_pdf": row.input_pdf,
                        "improved_ocr": row.improved_ocr,
                        "metadata": row.metadata,
                        "similarity": float(row.similarity),
                    }
                    for row in result
                ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to find similar documents: {str(e)}")
            raise DatabaseError(f"Failed to find similar documents: {str(e)}")

    @retry_with_backoff()
    def find_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Find a document by its ID."""
        try:
            with self.get_connection() as conn:
                query = text(
                    """
                    SELECT 
                        id, 
                        input_pdf,
                        output_base,
                        original_ocr,
                        improved_ocr,
                        embedding_vector,
                        metadata,
                        created_at,
                        updated_at
                    FROM documents
                    WHERE id = :id;
                """
                )

                result = conn.execute(query, {"id": doc_id}).fetchone()

                if result:
                    return {
                        "id": result.id,
                        "input_pdf": result.input_pdf,
                        "output_base": result.output_base,
                        "original_ocr": result.original_ocr,
                        "improved_ocr": result.improved_ocr,
                        "embedding_vector": result.embedding_vector,
                        "metadata": result.metadata,
                        "created_at": result.created_at,
                        "updated_at": result.updated_at,
                    }
                else:
                    return None

        except SQLAlchemyError as e:
            logger.error(f"Failed to find document by ID: {str(e)}")
            raise DatabaseError(f"Failed to find document by ID: {str(e)}")

    def close(self):
        """Close database connection pool and connector."""
        try:
            self.engine.dispose()
            self.connector.close()
            logger.info("Closed database connections")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
