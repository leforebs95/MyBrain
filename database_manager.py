from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import Json
import logging
from dataclasses import dataclass

from errors import DatabaseError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    host: str
    database: str
    user: str
    password: str
    port: int = 5432


class DatabaseManager:
    """
    Manages PostgreSQL database operations with support for vector storage.

    This class handles both traditional data storage and vector operations,
    serving as a complement to the Vector Store Manager.
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize database connection."""
        try:
            self.conn = psycopg2.connect(
                host=config.host,
                database=config.database,
                user=config.user,
                password=config.password,
                port=config.port,
            )
            self._init_database()
            logger.info("Initialized DatabaseManager")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _init_database(self):
        """Initialize database schema and extensions."""
        with self.conn.cursor() as cur:
            # Enable vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table
            cur.execute(
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

            # Create updated_at trigger
            cur.execute(
                """
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """
            )

            cur.execute(
                """
                DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """
            )

            self.conn.commit()

    @retry_with_backoff()
    def store_document(self, doc_data: Dict) -> str:
        """
        Store document data in PostgreSQL while simultaneously preparing JSONL for Vertex.AI.

        Args:
            doc_data: Dictionary containing document data

        Returns:
            str: Document ID
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (
                        id, input_pdf, output_base, original_ocr, 
                        improved_ocr, embedding_vector, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        input_pdf = EXCLUDED.input_pdf,
                        output_base = EXCLUDED.output_base,
                        original_ocr = EXCLUDED.original_ocr,
                        improved_ocr = EXCLUDED.improved_ocr,
                        embedding_vector = EXCLUDED.embedding_vector,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """,
                    (
                        doc_data["id"],
                        doc_data["input_pdf"],
                        doc_data["output_base"],
                        doc_data.get("original_ocr"),
                        doc_data.get("improved_ocr"),
                        doc_data.get("embedding"),
                        Json(doc_data.get("metadata", {})),
                    ),
                )
                self.conn.commit()
                return doc_data["id"]
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store document: {str(e)}")
            raise DatabaseError(f"Failed to store document: {str(e)}")

    @retry_with_backoff()
    def find_similar_documents(
        self, embedding: List[float], limit: int = 10
    ) -> List[Dict]:
        """
        Find similar documents using vector similarity search.

        Args:
            embedding: Query vector
            limit: Maximum number of results

        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        id, 
                        input_pdf,
                        improved_ocr,
                        1 - (embedding_vector <=> %s) as similarity
                    FROM documents
                    WHERE embedding_vector IS NOT NULL
                    ORDER BY embedding_vector <=> %s
                    LIMIT %s;
                """,
                    (embedding, embedding, limit),
                )

                results = []
                for row in cur.fetchall():
                    results.append(
                        {
                            "id": row[0],
                            "input_pdf": row[1],
                            "improved_ocr": row[2],
                            "similarity": float(row[3]),
                        }
                    )
                return results
        except Exception as e:
            logger.error(f"Failed to find similar documents: {str(e)}")
            raise DatabaseError(f"Failed to find similar documents: {str(e)}")

    def close(self):
        """Close database connection."""
        self.conn.close()
