from typing import List, Optional
import logging
from PyPDF2 import PdfReader, PdfWriter
import io
import os
from pathlib import Path

from errors import StorageError
from storage_manager import GCPStorageManager
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFReader:
    """
    Handles PDF reading, splitting, and uploading operations.

    This class provides functionality to:
    - Read PDF files from local filesystem or GCP storage
    - Split PDFs into individual pages
    - Upload split PDFs to GCP storage

    Attributes:
        storage_manager (GCPStorageManager): Instance of storage manager for GCP operations
    """

    def __init__(self, storage_manager: GCPStorageManager):
        """
        Initialize the PDF reader.

        Args:
            storage_manager (GCPStorageManager): Instance of storage manager
        """
        self.storage_manager = storage_manager
        logger.info("Initialized PDFReader")

    @retry_with_backoff()
    def read_pdf_from_gcp(self, bucket_name: str, blob_name: str) -> PdfReader:
        """
        Reads a PDF file from GCP Storage.

        Args:
            bucket_name (str): Name of the GCP bucket
            blob_name (str): Name of the blob (file path in bucket)

        Returns:
            PdfReader: PyPDF2 reader object

        Raises:
            StorageError: If reading the PDF fails
        """
        try:
            bucket = self.storage_manager.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            pdf_content = blob.download_as_bytes()

            return PdfReader(io.BytesIO(pdf_content))
        except Exception as e:
            logger.error(f"Failed to read PDF from GCP: {str(e)}")
            raise StorageError(f"Failed to read PDF from GCP: {str(e)}")

    @retry_with_backoff()
    def read_pdf_from_file(self, file_path: str) -> PdfReader:
        """
        Reads a PDF file from the local filesystem.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            PdfReader: PyPDF2 reader object

        Raises:
            StorageError: If reading the PDF fails
        """
        try:
            return PdfReader(file_path)
        except Exception as e:
            logger.error(f"Failed to read PDF from file: {str(e)}")
            raise StorageError(f"Failed to read PDF from file: {str(e)}")

    def split_pdf(self, pdf_reader: PdfReader) -> List[PdfWriter]:
        """
        Splits a PDF into individual pages.

        Args:
            pdf_reader (PdfReader): PyPDF2 reader object

        Returns:
            List[PdfWriter]: List of PDF writers, each containing one page

        Raises:
            StorageError: If splitting the PDF fails
        """
        try:
            writers = []
            for page_num in range(len(pdf_reader.pages)):
                writer = PdfWriter()
                writer.add_page(pdf_reader.pages[page_num])
                writers.append(writer)

            logger.info(f"Successfully split PDF into {len(writers)} pages")
            return writers
        except Exception as e:
            logger.error(f"Failed to split PDF: {str(e)}")
            raise StorageError(f"Failed to split PDF: {str(e)}")

    def write_page_to_gcp(
        self,
        writer: PdfWriter,
        bucket_name: str,
        output_prefix: str,
        page_number: int,
    ) -> str:
        """
        Writes a single PDF page to GCP Storage.

        Args:
            writer (PdfWriter): PyPDF2 writer containing the page
            bucket_name (str): Destination bucket name
            output_prefix (str): Prefix for the output file
            page_number (int): Page number for the filename

        Returns:
            str: Path of the uploaded file

        Raises:
            StorageError: If writing the PDF fails
        """
        try:
            # Create bytes buffer
            pdf_bytes = io.BytesIO()
            writer.write(pdf_bytes)
            pdf_bytes.seek(0)

            # Generate output path
            output_path = f"{output_prefix}_{page_number + 1}.pdf"

            # Upload to GCP
            bucket = self.storage_manager.storage_client.bucket(bucket_name)
            blob = bucket.blob(output_path)
            blob.upload_from_file(pdf_bytes, content_type="application/pdf")

            logger.info(f"Successfully wrote page {page_number + 1} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to write PDF page to GCP: {str(e)}")
            raise StorageError(f"Failed to write PDF page to GCP: {str(e)}")

    def process_and_split_pdf(
        self,
        input_bucket: str,
        input_path: str,
        output_bucket: str,
        output_prefix: str,
    ) -> List[str]:
        """
        Full pipeline to process a PDF: read from GCP, split into pages, and write back to GCP.

        Args:
            input_bucket (str): Source bucket name
            input_path (str): Path to source PDF in bucket
            output_bucket (str): Destination bucket name
            output_prefix (str): Prefix for output files

        Returns:
            List[str]: List of paths to the uploaded split PDFs

        Raises:
            StorageError: If any part of the process fails
        """
        try:
            # Read PDF from GCP
            pdf_reader = self.read_pdf_from_gcp(input_bucket, input_path)

            # Split into pages
            writers = self.split_pdf(pdf_reader)

            # Write pages back to GCP
            output_paths = []
            for i, writer in enumerate(writers):
                output_path = self.write_page_to_gcp(
                    writer, output_bucket, output_prefix, i
                )
                output_paths.append(output_path)

            logger.info(
                f"Successfully processed PDF {input_path} into {len(output_paths)} pages"
            )
            return output_paths
        except Exception as e:
            logger.error(f"Failed to process PDF {input_path}: {str(e)}")
            raise StorageError(f"Failed to process PDF: {str(e)}")

    def process_local_pdf(
        self, input_path: str, output_bucket: str, output_prefix: str
    ) -> List[str]:
        """
        Process a local PDF file: read from filesystem, split into pages, and write to GCP.

        Args:
            input_path (str): Path to local PDF file
            output_bucket (str): Destination bucket name
            output_prefix (str): Prefix for output files

        Returns:
            List[str]: List of paths to the uploaded split PDFs

        Raises:
            StorageError: If any part of the process fails
        """
        try:
            # Read PDF from local filesystem
            pdf_reader = self.read_pdf_from_file(input_path)

            # Split into pages
            writers = self.split_pdf(pdf_reader)

            # Write pages to GCP
            output_paths = []
            for i, writer in enumerate(writers):
                output_path = self.write_page_to_gcp(
                    writer, output_bucket, output_prefix, i
                )
                output_paths.append(output_path)

            logger.info(
                f"Successfully processed local PDF {input_path} into {len(output_paths)} pages"
            )
            return output_paths
        except Exception as e:
            logger.error(f"Failed to process local PDF {input_path}: {str(e)}")
            raise StorageError(f"Failed to process local PDF: {str(e)}")
