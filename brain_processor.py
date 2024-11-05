from typing import List, Dict, Optional, Tuple, NamedTuple
import json
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from time import sleep
import threading
from datetime import datetime

from storage_manager import GCPStorageManager
from ocr_processor import OCRProcessor
from claude_client import TextImprover
from embedding_generator import EmbeddingGenerator
from vector_store_manager import VectorStoreManager
from errors import BrainProcessingError, StorageError
from utils import RetryStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """
    Data class to store OCR processing results.

    Attributes:
        input_pdf (str): Path to the input PDF file
        output_base (str): Path to the output JSON file base uri
        original_ocr (str, optional): Raw OCR text
        improved_ocr (str, optional): Improved OCR text
        embedding (List[float], optional): Vector embedding of the improved text
    """

    input_pdf: str
    output_base: str
    original_ocr: Optional[str] = None
    improved_ocr: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class ProcessingProgress:
    """
    Tracks progress of batch processing operations.

    Attributes:
        total_jobs (int): Total number of jobs to process
        completed_jobs (int): Number of completed jobs
        failed_jobs (int): Number of failed jobs
        retried_jobs (int): Number of retried jobs
        start_time (datetime): When processing started
        end_time (Optional[datetime]): When processing completed
    """

    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    retried_jobs: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of processing."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100

    @property
    def duration(self) -> Optional[float]:
        """Calculate the duration of processing in seconds."""
        if not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict:
        """Convert progress to dictionary format."""
        return {
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "retried_jobs": self.retried_jobs,
            "success_rate": f"{self.success_rate:.2f}%",
            "duration": f"{self.duration:.2f}s" if self.duration else "ongoing",
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class ProcessingJob:
    """
    Data class to store document processing job information.

    Attributes:
        input_bucket (str): Input bucket name
        input_pdf (str): Input PDF path
        output_bucket (str): Output bucket name
        output_base (str): Base path for output files
    """

    input_bucket: str
    input_pdf: str
    output_bucket: str
    output_base: str


@dataclass
class ProcessingResult:
    """
    Stores the result of a processing attempt.

    Attributes:
        success (bool): Whether processing was successful
        result (Optional[OCRResult]): Processing result if successful
        error (Optional[str]): Error message if failed
        attempt (int): Which attempt this was
    """

    success: bool
    result: Optional[OCRResult] = None
    error: Optional[str] = None
    attempt: int = 1


class BrainProcessor:
    """Enhanced BrainProcessor with retry logic and progress tracking."""

    def __init__(
        self,
        config: Dict[str, str],
        max_workers: int = 4,
        retry_strategy: Optional[RetryStrategy] = None,
    ):
        """
        Initialize the brain processor.

        Args:
            config (Dict[str, str]): Configuration dictionary
            max_workers (int): Maximum number of parallel workers
            retry_strategy (RetryStrategy, optional): Retry strategy for failed operations
        """
        try:
            self.storage_manager = GCPStorageManager(
                project_id=config["gcp_project_id"]
            )
            self.ocr_processor = OCRProcessor(self.storage_manager)
            self.text_improver = TextImprover(config["anthropic_api_key"])
            self.embedding_generator = EmbeddingGenerator(config["voyage_api_key"])
            self.vector_store = VectorStoreManager(
                config["gcp_project_id"], config["gcp_location"]
            )
            self.max_workers = max_workers
            self.retry_strategy = retry_strategy or RetryStrategy()
            self._progress_lock = threading.Lock()
            logger.info(f"Initialized BrainProcessor with {max_workers} workers")
        except Exception as e:
            logger.error(f"Failed to initialize BrainProcessor: {str(e)}")
            raise BrainProcessingError(
                f"BrainProcessor initialization failed: {str(e)}"
            )

    def process_document_with_retry(
        self, job: ProcessingJob, progress: ProcessingProgress
    ) -> ProcessingResult:
        """
        Process a document with retry logic.

        Args:
            job (ProcessingJob): Processing job information
            progress (ProcessingProgress): Progress tracker

        Returns:
            ProcessingResult: Result of processing attempts
        """
        attempt = 1
        while attempt <= self.retry_strategy.max_retries:
            try:
                logger.info(f"Processing {job.input_pdf} (Attempt {attempt})")

                # Attempt processing
                gcs_source_uri = f"gs://{job.input_bucket}/{job.input_pdf}"
                gcs_destination_uri = f"gs://{job.output_bucket}/{job.output_base}"

                result = OCRResult(
                    input_pdf=job.input_pdf,
                    output_base=f"{job.output_base}",
                )

                self.ocr_processor.process_document(
                    gcs_source_uri, gcs_destination_uri, batch_size=1
                )
                for uri in self.ocr_processor.list_result_uris(
                    job.output_bucket, job.output_base
                ):
                    result.original_ocr += (
                        "\n"
                        + self.ocr_processor.extract_ocr_text_from_result(
                            bucket_name=job.output_bucket, extraction_prefix=uri
                        )
                    )
                result.improved_ocr = self.text_improver.improve_text(
                    result.original_ocr
                )
                result.embedding = self.embedding_generator.generate_embeddings(
                    [result.improved_ocr]
                )[0]

                # Update progress
                with self._progress_lock:
                    if attempt > 1:
                        progress.retried_jobs += 1
                    progress.completed_jobs += 1

                logger.info(
                    f"Successfully processed {job.input_pdf} on attempt {attempt}"
                )
                return ProcessingResult(success=True, result=result, attempt=attempt)

            except Exception as e:
                error_msg = (
                    f"Failed to process {job.input_pdf} (Attempt {attempt}): {str(e)}"
                )
                logger.warning(error_msg)

                if attempt == self.retry_strategy.max_retries:
                    with self._progress_lock:
                        progress.failed_jobs += 1
                    logger.error(f"All retry attempts failed for {job.input_pdf}")
                    return ProcessingResult(
                        success=False, error=error_msg, attempt=attempt
                    )

                # Wait before retrying
                sleep_time = self.retry_strategy.get_delay(attempt)
                logger.info(f"Retrying {job.input_pdf} in {sleep_time:.2f} seconds")
                sleep(sleep_time)
                attempt += 1

    def batch_process_documents(
        self,
        jobs: List[ProcessingJob],
        save_results: bool = True,
        output_prefix: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[OCRResult], ProcessingProgress]:
        """
        Processes multiple documents in parallel with progress tracking.

        Args:
            jobs (List[ProcessingJob]): List of processing jobs
            save_results (bool): Whether to save results to storage
            output_prefix (str, optional): Prefix for output files
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            Tuple[List[OCRResult], ProcessingProgress]: Successful results and progress info

        Raises:
            BrainProcessingError: If processing fails for any document
        """
        results = []
        failed_documents = []

        # Initialize progress tracking
        progress = ProcessingProgress(total_jobs=len(jobs))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_document_with_retry, job, progress): job
                for job in jobs
            }

            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                processing_result = future.result()

                if processing_result.success:
                    results.append(processing_result.result)
                else:
                    failed_documents.append((job.input_pdf, processing_result.error))

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress.to_dict())

        # Set completion time
        progress.end_time = datetime.now()

        # Save results if requested
        if save_results and results and output_prefix:
            try:
                self.save_results(results, output_prefix)
            except StorageError as e:
                logger.error(f"Failed to save some results: {str(e)}")

        # Report any failures
        if failed_documents:
            failure_messages = "\n".join(
                [f"{doc}: {err}" for doc, err in failed_documents]
            )
            logger.error(
                f"Batch processing completed with failures:\n{failure_messages}"
            )
            raise BrainProcessingError(
                f"Batch processing failed for some documents:\n{failure_messages}"
            )

        return results, progress

    def save_results(
        self,
        results: List[OCRResult],
        output_prefix: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, ProcessingProgress]:
        """
        Saves processing results to storage in parallel with retry logic.

        Args:
            results (List[OCRResult]): Processing results to save
            output_prefix (str): Prefix for output files
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            Tuple[bool, ProcessingProgress]: Success status and progress info

        Raises:
            StorageError: If saving results fails after all retries
        """
        failed_saves = []
        progress = ProcessingProgress(total_jobs=len(results))
        progress_lock = threading.Lock()

        def save_single_result(
            index: int, result: OCRResult, retry_count: int = 0
        ) -> Tuple[int, Optional[str]]:
            """Helper function to save a single result with retry logic."""
            while retry_count < self.retry_strategy.max_retries:
                try:
                    output_file = f"{output_prefix}_{index}.json"
                    result_dict = {
                        "input_pdf": result.input_pdf,
                        "output_base": result.output_base,
                        "original_ocr": result.original_ocr,
                        "improved_ocr": result.improved_ocr,
                        "embedding": result.embedding,
                    }

                    self.storage_manager.upload_data(
                        json.dumps(result_dict),
                        output_file,
                        self.storage_manager.vs_bucket,
                    )

                    with progress_lock:
                        if retry_count > 0:
                            progress.retried_jobs += 1
                        progress.completed_jobs += 1
                        if progress_callback:
                            progress_callback(progress.to_dict())

                    logger.info(
                        f"Saved results for {result.input_pdf} to {output_file}"
                        + (
                            f" (after {retry_count + 1} attempts)"
                            if retry_count > 0
                            else ""
                        )
                    )
                    return index, None

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Failed to save result {index} (Attempt {retry_count}): {str(e)}"
                    logger.warning(error_msg)

                    if retry_count < self.retry_strategy.max_retries:
                        sleep_time = self.retry_strategy.get_delay(retry_count)
                        logger.info(
                            f"Retrying save for index {index} in {sleep_time:.2f} seconds"
                        )
                        sleep(sleep_time)
                    else:
                        with progress_lock:
                            progress.failed_jobs += 1
                            if progress_callback:
                                progress_callback(progress.to_dict())
                        return index, error_msg

            return index, f"Exceeded maximum retry attempts for index {index}"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all save operations
            future_to_index = {
                executor.submit(save_single_result, i, result): i
                for i, result in enumerate(results)
            }

            # Process completed saves
            for future in as_completed(future_to_index):
                index, error = future.result()
                if error:
                    failed_saves.append((index, error))

        # Set completion time
        progress.end_time = datetime.now()

        # Handle any failures
        if failed_saves:
            failure_messages = "\n".join(
                [f"Result {idx}: {err}" for idx, err in failed_saves]
            )
            logger.error(
                f"Save operation completed with {len(failed_saves)} failures:\n{failure_messages}"
            )
            raise StorageError(f"Failed to save some results:\n{failure_messages}")

        return True, progress
