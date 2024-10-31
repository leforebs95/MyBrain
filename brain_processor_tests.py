from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import pytest
from datetime import datetime

from brain_processor import (
    BrainProcessor,
    OCRResult,
    ProcessingJob,
    ProcessingProgress,
    ProcessingResult,
    RetryStrategy,
)
from errors import BrainProcessingError, StorageError


@pytest.fixture
def config():
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
def retry_strategy():
    """Creates a test retry strategy."""
    return RetryStrategy(
        max_retries=2, initial_delay=0.1, max_delay=0.3, backoff_factor=2.0
    )


@pytest.fixture
def mock_components():
    """Creates mock components used by BrainProcessor."""
    with patch.multiple(
        "brain_processor",
        GCPStorageManager=Mock(),
        OCRProcessor=Mock(),
        TextImprover=Mock(),
        EmbeddingGenerator=Mock(),
        VectorStoreManager=Mock(),
    ) as mocks:
        yield mocks


@pytest.fixture
def brain_processor(config, retry_strategy, mock_components):
    """Creates a BrainProcessor instance with mock components."""
    processor = BrainProcessor(config, max_workers=2, retry_strategy=retry_strategy)

    # Setup default mock behaviors
    processor.ocr_processor.extract_text.return_value = "Raw text"
    processor.text_improver.improve_text.return_value = "Improved text"
    processor.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

    return processor


@pytest.fixture
def test_job():
    """Creates a test processing job."""
    return ProcessingJob(
        input_bucket="test-input",
        input_pdf="test.pdf",
        output_bucket="test-output",
        output_base="test_base",
    )


@pytest.fixture
def ocr_result():
    """Creates a test OCR result."""
    return OCRResult(
        input_pdf="test.pdf",
        output_json="test_output.json",
        original_ocr="Raw text",
        improved_ocr="Improved text",
        embedding=[0.1, 0.2, 0.3],
    )


class TestBrainProcessor:
    """Test suite for BrainProcessor class."""

    def test_initialization_success(self, config, retry_strategy, mock_components):
        """Test successful initialization of BrainProcessor."""
        processor = BrainProcessor(config, max_workers=2, retry_strategy=retry_strategy)

        assert isinstance(processor.storage_manager, Mock)
        assert isinstance(processor.ocr_processor, Mock)
        assert isinstance(processor.text_improver, Mock)
        assert isinstance(processor.embedding_generator, Mock)
        assert isinstance(processor.vector_store, Mock)
        assert processor.max_workers == 2
        assert processor.retry_strategy == retry_strategy

    def test_initialization_failure(self, config, retry_strategy):
        """Test initialization failure handling."""
        with patch(
            "brain_processor.GCPStorageManager", side_effect=Exception("Storage error")
        ):
            with pytest.raises(BrainProcessingError) as exc_info:
                BrainProcessor(config, retry_strategy=retry_strategy)
            assert "BrainProcessor initialization failed" in str(exc_info.value)

    def test_process_document_success_first_try(self, brain_processor, test_job):
        """Test successful document processing on first attempt."""
        progress = ProcessingProgress(total_jobs=1)
        result = brain_processor.process_document_with_retry(test_job, progress)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.attempt == 1
        assert result.result.input_pdf == "test.pdf"
        assert result.result.original_ocr == "Raw text"
        assert result.result.improved_ocr == "Improved text"
        assert result.result.embedding == [0.1, 0.2, 0.3]
        assert progress.completed_jobs == 1
        assert progress.failed_jobs == 0
        assert progress.retried_jobs == 0

    def test_process_document_success_after_retry(self, brain_processor, test_job):
        """Test successful document processing after retry."""
        brain_processor.ocr_processor.process_document.side_effect = [
            Exception("First try fails"),
            None,
        ]

        progress = ProcessingProgress(total_jobs=1)
        result = brain_processor.process_document_with_retry(test_job, progress)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.attempt == 2
        assert progress.completed_jobs == 1
        assert progress.failed_jobs == 0
        assert progress.retried_jobs == 1

    def test_process_document_failure_after_retries(self, brain_processor, test_job):
        """Test document processing failure after all retries."""
        brain_processor.ocr_processor.process_document.side_effect = Exception(
            "Processing error"
        )

        progress = ProcessingProgress(total_jobs=1)
        result = brain_processor.process_document_with_retry(test_job, progress)

        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert result.error is not None
        assert "Processing error" in str(result.error)
        assert result.attempt == brain_processor.retry_strategy.max_retries
        assert progress.completed_jobs == 0
        assert progress.failed_jobs == 1
        assert progress.retried_jobs == 0

    def test_batch_process_documents_success(self, brain_processor):
        """Test successful batch processing of documents."""
        jobs = [
            ProcessingJob(
                input_bucket="test-input",
                input_pdf=f"test{i}.pdf",
                output_bucket="test-output",
                output_base=f"test{i}_base",
            )
            for i in range(3)
        ]

        progress_updates = []

        def progress_callback(progress_info):
            progress_updates.append(progress_info)

        results, progress = brain_processor.batch_process_documents(
            jobs=jobs, save_results=False, progress_callback=progress_callback
        )

        assert len(results) == 3
        assert all(isinstance(r, OCRResult) for r in results)
        assert progress.total_jobs == 3
        assert progress.completed_jobs == 3
        assert progress.failed_jobs == 0
        assert len(progress_updates) > 0

    def test_batch_process_with_failures(self, brain_processor):
        """Test batch processing with some failures."""
        jobs = [
            ProcessingJob(
                input_bucket="test-input",
                input_pdf=f"test{i}.pdf",
                output_bucket="test-output",
                output_base=f"test{i}_base",
            )
            for i in range(3)
        ]

        # Track processing progress
        processed_jobs = 0
        failed_jobs = 0

        def mock_process(*args, **kwargs):
            nonlocal processed_jobs, failed_jobs
            job = args[0]
            if job.input_pdf == "test0.pdf":
                processed_jobs += 1
                return ProcessingResult(
                    success=True,
                    result=OCRResult(
                        input_pdf="test0.pdf",
                        output_json="test0_output.json",
                        original_ocr="Raw text",
                        improved_ocr="Improved text",
                        embedding=[0.1, 0.2, 0.3],
                    ),
                )
            else:
                failed_jobs += 1
                return ProcessingResult(success=False, error="Processing error")

        brain_processor.process_document_with_retry = Mock(side_effect=mock_process)

        try:
            brain_processor.batch_process_documents(jobs=jobs, save_results=False)
            pytest.fail("Expected BrainProcessingError was not raised")
        except BrainProcessingError as e:
            # Verify error message
            assert "Batch processing failed for some documents" in str(e)
            # Verify processing counts
            assert processed_jobs == 1  # One successful job
            assert failed_jobs == 2  # Two failed jobs
            # Verify all jobs were attempted
            assert brain_processor.process_document_with_retry.call_count == 3

    def test_save_results_success(self, brain_processor, ocr_result):
        """Test successful saving of results."""
        results = [ocr_result] * 3
        progress_updates = []

        def progress_callback(progress_info):
            progress_updates.append(progress_info)

        success, progress = brain_processor.save_results(
            results=results,
            output_prefix="test_batch",
            progress_callback=progress_callback,
        )

        assert success is True
        assert progress.total_jobs == 3
        assert progress.completed_jobs == 3
        assert progress.failed_jobs == 0
        assert len(progress_updates) > 0
        assert brain_processor.storage_manager.upload_data.call_count == 3

    def test_save_results_with_failures(self, brain_processor, ocr_result):
        """Test saving results with failures.

        The test verifies that:
        1. The function properly raises StorageError
        2. Progress tracking is accurate before the failure
        3. The error message contains expected information
        """
        results = [ocr_result] * 3
        progress_tracker = None

        # Mock upload_data to succeed twice then fail
        upload_calls = 0

        def mock_upload(*args, **kwargs):
            nonlocal upload_calls
            if upload_calls == 2:  # Fail on second upload
                raise Exception("Upload failed")
            upload_calls += 1

        brain_processor.storage_manager.upload_data.side_effect = mock_upload

        try:
            brain_processor.save_results(results=results, output_prefix="test_batch")
            pytest.fail("Expected StorageError was not raised")
        except StorageError as e:
            # Verify error message
            assert "Failed to save some results" in str(e)
            # Verify upload was attempted
            assert upload_calls == 2
            # Progress should show 1 successful upload before failure
            assert brain_processor.storage_manager.upload_data.call_count == 4
