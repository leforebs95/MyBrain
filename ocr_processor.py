from google.cloud import vision
import logging

from storage_manager import GCPStorageManager
from errors import OCRError
from utils import RetryStrategy, retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Handles OCR processing using Google Cloud Vision API.

    Attributes:
        storage_manager (GCPStorageManager): Storage manager instance
        vision_client (vision.ImageAnnotatorClient): Vision API client
    """

    def __init__(self, storage_manager: GCPStorageManager):
        """
        Initialize the OCR processor.

        Args:
            storage_manager (GCPStorageManager): Instance of storage manager
        """
        self.storage_manager = storage_manager
        try:
            self.vision_client = vision.ImageAnnotatorClient()
            logger.info("Initialized Vision API client")
        except Exception as e:
            logger.error(f"Failed to initialize Vision API client: {str(e)}")
            raise OCRError(f"Vision API client initialization failed: {str(e)}")

    @retry_with_backoff(RetryStrategy(max_retries=2, backoff_factor=2))
    def process_document(
        self, gcs_source_uri: str, gcs_destination_uri: str, batch_size: int = 2
    ) -> None:
        """
        Processes a PDF document using Google Cloud Vision OCR.

        Args:
            gcs_source_uri (str): URI of the source PDF in GCS
            gcs_destination_uri (str): URI for the OCR output in GCS
            batch_size (int): Number of pages to process in each batch

        Raises:
            OCRError: If the OCR operation fails
        """
        try:
            feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            gcs_source = vision.GcsSource(uri=gcs_source_uri)
            input_config = vision.InputConfig(
                gcs_source=gcs_source, mime_type="application/pdf"
            )

            gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
            output_config = vision.OutputConfig(
                gcs_destination=gcs_destination, batch_size=batch_size
            )

            request = vision.AsyncAnnotateFileRequest(
                features=[feature],
                input_config=input_config,
                output_config=output_config,
            )

            operation = self.vision_client.async_batch_annotate_files(
                requests=[request]
            )
            logger.info(f"Started OCR processing for {gcs_source_uri}")

            result = operation.result(timeout=420)
            logger.info(f"Completed OCR processing for {gcs_source_uri}")
            return result

        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise OCRError(f"OCR processing failed: {str(e)}")

    def list_result_uris(self, bucket_name: str, uri_pattern: str) -> list:
        """Lists all the URIs in a GCP bucket that match a given pattern."""
        try:
            # Get the bucket
            bucket = self.storage_manager.storage_client.bucket(bucket_name)

            # List all the blobs in the bucket
            blobs = bucket.list_blobs()

            # Filter blobs that match the pattern
            matching_uris = [blob.name for blob in blobs if uri_pattern in blob.name]
            logger.info(
                f"Found {len(matching_uris)} matching URIs in bucket '{bucket_name}'"
            )
            return matching_uris
        except Exception as e:
            logger.error(f"Failed to list result URIs: {str(e)}")
            raise OCRError(f"Failed to list result URIs: {str(e)}")

    def extract_ocr_text_from_result(
        self, bucket_name: str, extraction_prefix: str
    ) -> str:
        """
        Extracts text from a JSON file stored in a Google Cloud Storage bucket.
        This function downloads a JSON file from the specified bucket and extraction prefix,
        parses the JSON content, and extracts the full text annotations from the responses.
        Args:
            bucket_name (str): The name of the Google Cloud Storage bucket.
            extraction_prefix (str): The prefix (path) to the JSON file within the bucket.
        Returns:
            str: The extracted text from the JSON file.
        """
        import json

        try:
            # Get the bucket
            bucket = self.storage_manager.storage_client.bucket(bucket_name)

            # Get the blob (file) from the bucket
            blob = bucket.blob(extraction_prefix)

            # Download the file content as a string
            json_content = blob.download_as_text()
            to_json = json.loads(json_content)
            text = "\n".join(
                [resp["fullTextAnnotation"]["text"] for resp in to_json["responses"]]
            )
            logger.info(f"Successfully extracted OCR text from result")
            return text
        except Exception as e:
            logger.error(f"Failed to extract OCR text from result: {str(e)}")
            raise OCRError(f"Failed to extract OCR text from result: {str(e)}")
