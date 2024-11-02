from unittest.mock import Mock, patch
import pytest

from storage_manager import GCPStorageManager
from ocr_processor import OCRProcessor
from claude_client import TextImprover, ClaudeResponse
from embedding_generator import EmbeddingGenerator
from brain_processor import BrainProcessor, OCRResult
from errors import BrainProcessingError, StorageError, OCRError, EmbeddingError


class TestGCPStorageManager:
    """Test suite for GCPStorageManager class."""

    @pytest.fixture
    def storage_manager(self):
        """Creates a GCPStorageManager instance with mock client."""
        with patch("google.cloud.storage.Client") as mock_client:
            manager = GCPStorageManager(
                project_id="test-project",
            )
            manager.storage_client = mock_client
            return manager

    def test_initialization(self):
        """Test successful initialization of GCPStorageManager."""
        with patch("google.cloud.storage.Client") as mock_client:
            manager = GCPStorageManager(
                project_id="test-project",
            )
            assert manager.project_id == "test-project"
            mock_client.assert_called_once_with(project="test-project")

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch(
            "google.cloud.storage.Client", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(StorageError) as exc_info:
                GCPStorageManager(
                    project_id="test-project",
                )
            assert "Storage client initialization failed" in str(exc_info.value)

    def test_list_blobs(self, storage_manager):
        """Test listing blobs from a bucket."""
        mock_bucket = Mock()
        mock_blobs = [Mock(name="blob1"), Mock(name="blob2")]
        mock_bucket.list_blobs.return_value = mock_blobs
        storage_manager.storage_client.bucket.return_value = mock_bucket

        blobs = storage_manager.list_blobs("test-bucket", "test-prefix")

        assert blobs == mock_blobs
        storage_manager.storage_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.list_blobs.assert_called_once_with(prefix="test-prefix")

    def test_list_blobs_failure(self, storage_manager):
        """Test failure handling when listing blobs."""
        storage_manager.storage_client.bucket.side_effect = Exception(
            "Bucket access denied"
        )

        with pytest.raises(StorageError) as exc_info:
            storage_manager.list_blobs("test-bucket", "test-prefix")
        assert "Failed to list blobs" in str(exc_info.value)

    def test_upload_json(self, storage_manager):
        """Test JSON upload functionality."""
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        storage_manager.storage_client.bucket.return_value = mock_bucket

        result = storage_manager.upload_data(
            data='{"test": "data"}',
            file_name="test.json",
            bucket_name="test-bucket",
        )

        assert result == "test.json"
        mock_blob.upload_from_string.assert_called_once_with(
            '{"test": "data"}', content_type="application/json"
        )


class TestOCRProcessor:
    """Test suite for OCRProcessor class."""

    @pytest.fixture
    def ocr_processor(self):
        """Creates an OCRProcessor instance with mock clients."""
        mock_storage_manager = Mock()
        with patch("google.cloud.vision.ImageAnnotatorClient") as mock_vision:
            processor = OCRProcessor(mock_storage_manager)
            processor.vision_client = mock_vision
            return processor

    def test_initialization(self):
        """Test successful initialization of OCRProcessor."""
        mock_storage_manager = Mock()
        with patch("google.cloud.vision.ImageAnnotatorClient") as mock_vision:
            processor = OCRProcessor(mock_storage_manager)
            assert processor.storage_manager == mock_storage_manager
            mock_vision.assert_called_once()

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        mock_storage_manager = Mock()
        with patch(
            "google.cloud.vision.ImageAnnotatorClient",
            side_effect=Exception("Vision API error"),
        ):
            with pytest.raises(OCRError) as exc_info:
                OCRProcessor(mock_storage_manager)
            assert "Vision API client initialization failed" in str(exc_info.value)

    def test_process_document(self, ocr_processor):
        """Test document processing functionality."""
        mock_operation = Mock()
        mock_operation.result.return_value = None
        ocr_processor.vision_client.async_batch_annotate_files.return_value = (
            mock_operation
        )

        ocr_processor.process_document(
            gcs_source_uri="gs://test-bucket/test.pdf",
            gcs_destination_uri="gs://test-bucket/output/",
        )

        ocr_processor.vision_client.async_batch_annotate_files.assert_called_once()
        mock_operation.result.assert_called_once_with(timeout=420)

    def test_process_document_failure(self, ocr_processor):
        """Test failure handling in document processing."""
        ocr_processor.vision_client.async_batch_annotate_files.side_effect = Exception(
            "OCR failed"
        )

        with pytest.raises(OCRError) as exc_info:
            ocr_processor.process_document(
                gcs_source_uri="gs://test-bucket/test.pdf",
                gcs_destination_uri="gs://test-bucket/output/",
            )
        assert "OCR processing failed" in str(exc_info.value)


class TestTextImprover:
    """Test suite for TextImprover class."""

    @pytest.fixture
    def text_improver(self):
        """Creates a TextImprover instance with mock client."""
        with patch("claude_client.ClaudeClient") as mock_claude:
            improver = TextImprover("test-k[ey")
            improver.claude_client = mock_claude
            return improver

    def test_initialization(self):
        """Test successful initialization of TextImprover."""
        with patch("claude_client.ClaudeClient") as mock_claude:
            improver = TextImprover("test-key")
            mock_claude.assert_called_once_with("test-key")
            assert isinstance(improver.SYSTEM_MESSAGE, str)

    def test_initialization_with_named_parameter(self):
        """Test initialization with named parameter."""
        with patch("claude_client.ClaudeClient") as mock_claude:
            improver = TextImprover(api_key="test-key")
            # Alternative test for named parameter
            mock_claude.assert_called_once_with("test-key")
            assert isinstance(improver.SYSTEM_MESSAGE, str)

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch(
            "claude_client.ClaudeClient",
            side_effect=BrainProcessingError("Claude client error"),
        ):
            with pytest.raises(BrainProcessingError) as exc_info:
                TextImprover("test-key")
            assert "Claude client error" in str(exc_info.value)

    def test_improve_text(self, text_improver):
        """Test text improvement functionality."""
        mock_response = Mock(spec=ClaudeResponse)
        mock_response.content = "Improved text"
        text_improver.claude_client.process_text.return_value = mock_response

        result = text_improver.improve_text("Raw text")

        assert result == "Improved text"
        text_improver.claude_client.process_text.assert_called_once_with(
            text="Raw text", system_message=TextImprover.SYSTEM_MESSAGE
        )

    def test_improve_text_failure(self, text_improver):
        """Test failure handling in text improvement."""
        text_improver.claude_client.process_text.side_effect = BrainProcessingError(
            "Processing failed"
        )

        with pytest.raises(BrainProcessingError) as exc_info:
            text_improver.improve_text("Raw text")
        assert "Text improvement failed" in str(exc_info.value)


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    @pytest.fixture
    def embedding_generator(self):
        """Creates an EmbeddingGenerator instance with mock client."""
        with patch("voyageai.Client") as mock_voyage:
            generator = EmbeddingGenerator("test-key")
            generator.client = mock_voyage
            return generator

    def test_initialization(self):
        """Test successful initialization of EmbeddingGenerator."""
        with patch("voyageai.Client") as mock_voyage:
            generator = EmbeddingGenerator("test-key")
            mock_voyage.assert_called_once_with(api_key="test-key")

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch("voyageai.Client", side_effect=Exception("Voyage AI API error")):
            with pytest.raises(EmbeddingError) as exc_info:
                EmbeddingGenerator("test-key")
            assert "Voyage AI client initialization failed" in str(exc_info.value)

    def test_generate_embeddings(self, embedding_generator):
        """Test embedding generation functionality."""
        mock_result = Mock()
        mock_result.embeddings = [[0.1, 0.2, 0.3]]
        embedding_generator.client.embed.return_value = mock_result

        result = embedding_generator.generate_embeddings(["Test text"])

        assert result == [[0.1, 0.2, 0.3]]
        embedding_generator.client.embed.assert_called_once_with(
            ["Test text"], model="voyage-3", input_type="document"
        )

    def test_generate_embeddings_failure(self, embedding_generator):
        """Test failure handling in embedding generation."""
        embedding_generator.client.embed.side_effect = Exception("Embedding failed")

        with pytest.raises(EmbeddingError) as exc_info:
            embedding_generator.generate_embeddings(["Test text"])
        assert "Embedding generation failed" in str(exc_info.value)


class TestBrainProcessor:
    """Test suite for BrainProcessor class."""

    @pytest.fixture
    def config(self):
        """Creates a test configuration dictionary."""
        return {
            "gcp_project_id": "test-project",
            "gcp_location": "us-central1",
            "brain_bucket": "test-brain",
            "vs_bucket": "test-vs",
            "anthropic_api_key": "test-anthropic-key",
            "voyage_api_key": "test-voyage-key",
        }

    @pytest.fixture
    def brain_processor(self, config):
        """Creates a BrainProcessor instance with mock components."""
        with patch.multiple(
            "brain_processor",
            GCPStorageManager=Mock(),
            OCRProcessor=Mock(),
            TextImprover=Mock(),
            EmbeddingGenerator=Mock(),
            VectorStoreManager=Mock(),
        ):
            processor = BrainProcessor(config)
            return processor

    def test_initialization(self, config):
        """Test successful initialization of BrainProcessor."""
        with patch.multiple(
            "brain_processor",
            GCPStorageManager=Mock(),
            OCRProcessor=Mock(),
            TextImprover=Mock(),
            EmbeddingGenerator=Mock(),
            VectorStoreManager=Mock(),
        ):
            processor = BrainProcessor(config)
            assert isinstance(processor.storage_manager, Mock)
            assert isinstance(processor.ocr_processor, Mock)
            assert isinstance(processor.text_improver, Mock)
            assert isinstance(processor.embedding_generator, Mock)
            assert isinstance(processor.vector_store, Mock)

    def test_initialization_failure(self, config):
        """Test initialization failure handling."""
        with patch(
            "brain_processor.GCPStorageManager",
            side_effect=Exception("Storage error"),
        ):
            with pytest.raises(BrainProcessingError) as exc_info:
                BrainProcessor(config)
            assert "BrainProcessor initialization failed" in str(exc_info.value)

    def test_process_document(self, brain_processor):
        """Test document processing pipeline."""
        # Mock component responses
        brain_processor.ocr_processor.extract_text.return_value = "Raw text"
        brain_processor.text_improver.improve_text.return_value = "Improved text"
        brain_processor.embedding_generator.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3]
        ]

        result = brain_processor.process_document("test.pdf")

        assert isinstance(result, OCRResult)
        assert result.input_pdf == "test.pdf"
        assert result.original_ocr == "Raw text"
        assert result.improved_ocr == "Improved text"
        assert result.embedding == [0.1, 0.2, 0.3]

    def test_process_document_failure(self, brain_processor):
        """Test failure handling in document processing pipeline."""
        brain_processor.ocr_processor.process_document.side_effect = Exception(
            "Pipeline error"
        )

        with pytest.raises(BrainProcessingError) as exc_info:
            brain_processor.process_document("test.pdf")
        assert "Document processing failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
