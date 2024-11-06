from typing import List
from google.cloud import aiplatform, aiplatform_v1
from google.cloud import exceptions as google_exceptions
import uuid
import base64
import logging

from errors import VectorStoreError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations using Google Cloud Vertex AI.

    Attributes:
        project_id (str): Google Cloud project ID
        location (str): Google Cloud region
    """

    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the vector store manager.

        Args:
            project_id (str): Google Cloud project ID
            location (str): Google Cloud region
        """
        try:
            self.project_id = project_id
            self.location = location
            aiplatform.init(project=project_id, location=location)
            logger.info(f"Initialized Vector Store Manager for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store Manager: {str(e)}")
            raise VectorStoreError(
                f"Vector Store Manager initialization failed: {str(e)}"
            )

    @retry_with_backoff()
    def get_index(self, index_id: str) -> aiplatform.MatchingEngineIndex:
        """
        Retrieves a vector store index.

        Args:
            index_id (str): ID of the index to retrieve

        Returns:
            aiplatform.MatchingEngineIndex: Retrieved index

        Raises:
            VectorStoreError: If index retrieval fails
        """
        try:
            index = aiplatform.MatchingEngineIndex(index_id)
            logger.info(f"Retrieved index {index_id}")
            return index
        except google_exceptions.NotFound as e:
            logger.error(f"Index {index_id} not found: {str(e)}")
            raise VectorStoreError(f"Vector index {index_id} not found")
        except Exception as e:
            logger.error(f"Failed to get index: {str(e)}")
            raise VectorStoreError(f"Failed to get index: {str(e)}")

    @retry_with_backoff()
    def find_neighbors(
        self,
        api_endpoint: str,
        feature_vector: List[float],
        endpoint_id: str,
        deployment_id: str,
        neighbor_count: int = 10,
    ) -> aiplatform_v1.types.FindNeighborsResponse:
        """
        Finds nearest neighbors for a given vector.

        Args:
            api_endpoint (str): Vector search API endpoint
            feature_vector (List[float]): Query vector
            endpoint_id (str): ID of the endpoint to query
            deployment_id (str): ID of the deployed index
            neighbor_count (int): Number of neighbors to retrieve

        Returns:
            aiplatform_v1.types.FindNeighborsResponse: Search results

        Raises:
            VectorStoreError: If neighbor search fails
        """
        try:
            client = aiplatform_v1.MatchServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )

            full_endpoint_id = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{endpoint_id}"

            datapoint = aiplatform_v1.IndexDatapoint(feature_vector=feature_vector)
            query = aiplatform_v1.FindNeighborsRequest.Query(
                datapoint=datapoint, neighbor_count=neighbor_count
            )

            request = aiplatform_v1.FindNeighborsRequest(
                index_endpoint=full_endpoint_id,
                deployed_index_id=deployment_id,
                queries=[query],
                return_full_datapoint=False,
            )

            response = client.find_neighbors(request)
            logger.info(f"Successfully found {neighbor_count} neighbors")
            return response

        except Exception as e:
            logger.error(f"Failed to find neighbors: {str(e)}")
            raise VectorStoreError(f"Failed to find neighbors: {str(e)}")
