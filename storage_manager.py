from typing import List
import logging

from utils import retry_with_backoff
from google.cloud import storage

from errors import StorageError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BrainStorageManager:
    """
    Manages Google Cloud Storage operations for the brain processing pipeline.

    Attributes:
        project_id (str): Google Cloud project ID
        brain_bucket (str): Name of the bucket storing original documents
        vs_bucket (str): Name of the bucket storing vector store data
        storage_client (storage.Client): Google Cloud Storage client
    """

    def __init__(self, project_id: str, brain_bucket: str, vs_bucket: str):
        """
        Initialize the storage manager.

        Args:
            project_id (str): Google Cloud project ID
            brain_bucket (str): Name of the brain storage bucket
            vs_bucket (str): Name of the vector store bucket
        """
        self.project_id = project_id
        self.brain_bucket = brain_bucket
        self.vs_bucket = vs_bucket
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
    def upload_json(self, json_data: str, file_name: str, bucket_name: str) -> str:
        """
        Uploads JSON data to a specified GCP bucket.

        Args:
            json_data (str): JSON data to upload
            file_name (str): Destination file name
            bucket_name (str): Destination bucket name

        Returns:
            str: Name of the uploaded file

        Raises:
            StorageError: If the upload fails
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            blob.upload_from_string(json_data, content_type="application/json")
            logger.info(f"Successfully uploaded {file_name} to {bucket_name}")
            return file_name
        except Exception as e:
            logger.error(f"Failed to upload JSON: {str(e)}")
            raise StorageError(f"Failed to upload JSON: {str(e)}")
