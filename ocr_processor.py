from google.cloud import vision
import logging

from storage_manager import GCPStorageManager
from errors import OCRError
from utils import retry_with_backoff

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

    @retry_with_backoff(retries=2, backoff_in_seconds=2)
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

            operation.result(timeout=420)
            logger.info(f"Completed OCR processing for {gcs_source_uri}")

        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise OCRError(f"OCR processing failed: {str(e)}")
