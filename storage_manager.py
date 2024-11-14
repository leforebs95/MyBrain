from typing import List
import logging

from utils import RetryStrategy, retry_with_backoff
from google.cloud import storage

from errors import StorageError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GCPStorageManager:
    """
    Manages Google Cloud Storage operations for the brain processing pipeline.

    Attributes:
        project_id (str): Google Cloud project ID
    """

    def __init__(self, project_id: str):
        """
        Initialize the storage manager.

        Args:
            project_id (str): Google Cloud project ID
        """
        self.project_id = project_id
        try:
            self.storage_client = storage.Client(project=project_id)
            logger.info(f"Initialized storage client for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {str(e)}")
            raise StorageError(f"Storage client initialization failed: {str(e)}")

    @retry_with_backoff()
    def list_blobs(self, bucket_name: str, prefix: str) -> List[storage.Blob]:
        """
        Lists all files in a GCP bucket with a given prefix.

        Args:
            bucket_name (str): Name of the bucket
            prefix (str): Prefix to filter blobs

        Returns:
            List[storage.Blob]: List of matching blobs

        Raises:
            StorageError: If the operation fails
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            logger.info(
                f"Listed {len(blobs)} blobs in {bucket_name} with prefix {prefix}"
            )
            return blobs
        except Exception as e:
            logger.error(f"Failed to list blobs: {str(e)}")
            raise StorageError(f"Failed to list blobs: {str(e)}")

    @retry_with_backoff()
    def upload_data(
        self,
        data: str,
        prefix: str,
        bucket_name: str,
        content_type: str = "application/json",
    ) -> str:
        """
        Uploads data to a specified GCP bucket.

        Args:
            data (str): Data to upload
            prefix (str): Destination file name
            bucket_name (str): Destination bucket name

        Returns:
            str: Name of the uploaded file

        Raises:
            StorageError: If the upload fails
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(prefix)
            blob.upload_from_string(data, content_type=content_type)
            logger.info(f"Successfully uploaded {prefix} to {bucket_name}")
            return prefix
        except Exception as e:
            logger.error(f"Failed to upload JSON: {str(e)}")
            raise StorageError(f"Failed to upload JSON: {str(e)}")

    @retry_with_backoff()
    def upload_file(
        self,
        file_path: str,
        prefix: str,
        bucket_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Uploads a file to a specified GCP bucket.

        Args:
            file_path (str): Path to the file to upload
            prefix (str): Destination file name
            bucket_name (str): Destination bucket name

        Returns:
            str: Name of the uploaded file

        Raises:
            StorageError: If the upload fails
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(prefix)
            blob.upload_from_filename(file_path, content_type=content_type)
            logger.info(f"Successfully uploaded {prefix} to {bucket_name}")
            return prefix
        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            raise StorageError(f"Failed to upload file: {str(e)}")

    @retry_with_backoff(RetryStrategy(max_retries=0))
    def download_data(self, bucket_name: str, blob_name: str) -> str:
        """
        Downloads a blob from a specified GCP bucket as text.

        Args:
            bucket_name (str): Name of the bucket
            blob_name (str): Name of the blob to download

        Returns:
            str: Content of the downloaded blob

        Raises:
            StorageError: If the download fails
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            data = blob.download_as_text()
            logger.info(f"Successfully downloaded {blob_name} from {bucket_name}")
            return data
        except Exception as e:
            logger.error(f"Failed to download blob: {str(e)}")
            raise StorageError(f"Failed to download blob: {str(e)}")
