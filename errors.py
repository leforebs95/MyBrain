class BrainProcessingError(Exception):
    """Base exception class for brain processing errors."""

    pass


class StorageError(BrainProcessingError):
    """Raised when storage operations fail."""

    pass


class OCRError(BrainProcessingError):
    """Raised when OCR operations fail."""

    pass


class EmbeddingError(BrainProcessingError):
    """Raised when embedding operations fail."""

    pass


class VectorStoreError(BrainProcessingError):
    """Raised when vector store operations fail."""

    pass


class DatabaseError(BrainProcessingError):
    """Raised when database operations fail."""

    pass
