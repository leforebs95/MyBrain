from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import insert, JSONB
from pgvector.sqlalchemy import Vector
from google.cloud.sql.connector import Connector

from brain_processor import OCRResult
from errors import DatabaseError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()


@dataclass
class DatabaseConfig:
    instance_connection_name: str  # e.g., "project:region:instance"
    database: str
    user: str
    password: str
    pool_size: int = 5
    max_overflow: int = 2


class Document(Base):
    """SQLAlchemy model for documents table."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    input_pdf = Column(String, nullable=False)
    output_base = Column(String, nullable=False)
    original_ocr = Column(String)
    improved_ocr = Column(String)
    embedding = Column(Vector(1024))
    doc_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> Dict:
        """Convert model instance to dictionary."""
        return {
            "id": self.id,
            "input_pdf": self.input_pdf,
            "output_base": self.output_base,
            "original_ocr": self.original_ocr,
            "improved_ocr": self.improved_ocr,
            "embedding": (list(self.embedding) if self.embedding else None),
            "doc_metadata": self.doc_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatabaseManager:
    """Manages PostgreSQL database operations using SQLAlchemy ORM."""

    def __init__(self, config: DatabaseConfig):
        """Initialize database connection pool and ORM setup."""
        try:
            self.config = config
            self.connector = Connector()

            # Initialize the connection pool with SQLAlchemy
            self.engine = create_engine(
                "postgresql+pg8000://",
                creator=self._get_connection,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
            )

            # Create session factory
            self.Session = sessionmaker(bind=self.engine)

            # Initialize database
            self._init_database()

            logger.info("Initialized DatabaseManager with SQLAlchemy ORM")
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
    def get_session(self) -> Session:
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _init_database(self):
        """Initialize database schema and extensions."""
        try:
            # Create vector extension if it doesn't exist
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

            # Create all tables
            Base.metadata.create_all(self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    @retry_with_backoff()
    def store_document(self, doc_data: Dict) -> str:
        """Store or update a document in the database."""
        try:
            with self.get_session() as session:
                document = Document(
                    id=doc_data["id"],
                    input_pdf=doc_data["input_pdf"],
                    output_base=doc_data["output_base"],
                    original_ocr=doc_data.get("original_ocr"),
                    improved_ocr=doc_data.get("improved_ocr"),
                    embedding=doc_data.get("embedding"),
                    doc_metadata=doc_data.get("doc_metadata", {}),
                )

                session.merge(
                    document
                )  # Use merge instead of add to handle both insert and update
                return document.id

        except Exception as e:
            logger.error(f"Failed to store document: {str(e)}")
            raise DatabaseError(f"Failed to store document: {str(e)}")

    @retry_with_backoff()
    def bulk_update_documents(self, ocr_results: List[OCRResult]) -> List[str]:
        """Perform bulk update of documents."""
        try:
            with self.get_session() as session:
                documents = []
                for result in ocr_results:
                    document = dict(
                        id=result.id,
                        input_pdf=result.input_pdf,
                        output_base=result.output_base,
                        original_ocr=result.original_ocr,
                        improved_ocr=result.improved_ocr,
                        embedding=result.embedding,
                        doc_metadata=result.doc_metadata,
                    )
                    documents.append(document)

                # Create the upsert statement
                stmt = insert(Document).values(documents)

                # Handle the ON CONFLICT case
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "input_pdf": stmt.excluded.input_pdf,
                        "output_base": stmt.excluded.output_base,
                        "original_ocr": stmt.excluded.original_ocr,
                        "improved_ocr": stmt.excluded.improved_ocr,
                        "embedding": stmt.excluded.embedding,
                        "doc_metadata": stmt.excluded.doc_metadata,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )

                # Execute the statement
                result = session.execute(stmt)
                session.commit()

                # Return list of document IDs
                return [doc["id"] for doc in documents]

        except Exception as e:
            logger.error(f"Bulk update operation failed: {str(e)}")
            raise DatabaseError(f"Bulk update operation failed: {str(e)}")

    @retry_with_backoff()
    def find_similar_documents(
        self, embedding: List[float], limit: int = 10
    ) -> List[Dict]:
        """Find similar documents using vector similarity search."""
        try:
            with self.get_session() as session:
                # Using SQLAlchemy's text() for the vector operation
                results = (
                    session.query(Document)
                    .order_by(Document.embedding.cosine_distance(embedding))
                    .limit(limit)
                    .all()
                )

                return [doc.to_dict() for doc in results]

        except Exception as e:
            logger.error(f"Failed to find similar documents: {str(e)}")
            raise DatabaseError(f"Failed to find similar documents: {str(e)}")

    @retry_with_backoff()
    def find_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Find a document by its ID."""
        try:
            with self.get_session() as session:
                document = session.query(Document).filter(Document.id == doc_id).first()
                return document.to_dict() if document else None

        except Exception as e:
            logger.error(f"Failed to find document by ID: {str(e)}")
            raise DatabaseError(f"Failed to find document by ID: {str(e)}")

    @retry_with_backoff()
    def get_table_as_list(self, table_name: str) -> List[Dict]:
        """
        Query all records from a table and return them as a list of dictionaries.

        Args:
            table_name (str): Name of the table to query

        Returns:
            List[Dict]: List of records as dictionaries

        Example:
            docs = db.get_table_as_list('documents')
        """
        try:
            with self.get_session() as session:
                # Get table model from table name
                table = inspect(self.engine).get_table_names()
                if table_name not in table:
                    raise DatabaseError(f"Table '{table_name}' not found in database")

                # Build and execute query
                query = text(f"SELECT * FROM {table_name}")
                result = session.execute(query)

                # Convert results to list of dictionaries
                return [dict(row._mapping) for row in result]

        except Exception as e:
            logger.error(f"Failed to get records from {table_name}: {str(e)}")
            raise DatabaseError(f"Failed to get records from {table_name}: {str(e)}")

    def close(self):
        """Close database connection pool and connector."""
        try:
            self.engine.dispose()
            self.connector.close()
            logger.info("Closed database connections")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
