from dataclasses import dataclass
from typing import List, Optional
from google.cloud import aiplatform, aiplatform_v1
from google.cloud import exceptions as google_exceptions
import uuid
import base64
import logging

from ulid import ULID

from errors import VectorStoreError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    project_id: str
    api_endpoint: str
    location: str


class VectorStoreManager:
    """
    Manages vector store operations using Google Cloud Vertex AI.

    Attributes:
        project_id (str): Google Cloud project ID
        location (str): Google Cloud region
    """

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store manager.

        Args:
            project_id (str): Google Cloud project ID
            location (str): Google Cloud region
        """
        try:
            self.project_id = config.project_id
            self.location = config.location
            self.api_endpoint = config.api_endpoint
            aiplatform.init(project=config.project_id, location=config.location)
            logger.info(
                f"Initialized Vector Store Manager for project {config.project_id}"
            )
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
    def update_index(
        self,
        index_id: str,
        bucket: str,
        prefix: str,
        is_complete_overwrite: Optional[bool] = None,
    ) -> aiplatform.MatchingEngineIndex:
        """
        Updates an existing vector store index with new data from a GCS bucket.

        Args:
            index_id (str): ID of the index to update
            bucket (str): GCS bucket containing the new data
            prefix (str): Prefix of the new data files in the GCS bucket

        Returns:
            aiplatform.MatchingEngineIndex: Updated index

        Raises:
            VectorStoreError: If index update fails
        """
        gcs_uri = f"gs://{bucket}/{prefix}"
        try:
            index = self.get_index(index_id)
            index.update_embeddings(
                contents_delta_uri=gcs_uri, is_complete_overwrite=is_complete_overwrite
            )
            logger.info(
                f"Successfully updated index {index_id} with data from gs://{bucket}/{prefix}"
            )
            return index

        except Exception as e:
            logger.error(f"Failed to update index: {str(e)}")
            raise VectorStoreError(f"Failed to update index: {str(e)}")

    @retry_with_backoff()
    def get_index_endpoint(
        self, endpoint_id: str
    ) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Retrieves a vector store index endpoint.

        Args:
            endpoint_id (str): ID of the endpoint to retrieve

        Returns:
            aiplatform.MatchingEngineIndexEndpoint: Retrieved endpoint

        Raises:
            VectorStoreError: If endpoint retrieval fails
        """
        try:
            endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_id)
            logger.info(f"Retrieved endpoint {endpoint_id}")
            return endpoint
        except google_exceptions.NotFound as e:
            logger.error(f"Endpoint {endpoint_id} not found: {str(e)}")
            raise VectorStoreError(f"Vector endpoint {endpoint_id} not found")
        except Exception as e:
            logger.error(f"Failed to get endpoint: {str(e)}")
            raise VectorStoreError(f"Failed to get endpoint: {str(e)}")

    @retry_with_backoff()
    def create_index_deployment(
        self,
        index_id: str,
        endpoint_id: str,
        deployed_index_name: str,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
    ) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Creates a new deployment of a vector index to an endpoint.

        Args:
            index_id (str): ID of the index to deploy
            endpoint_id (str): ID of the endpoint to deploy to
            deployed_index_id (str): Custom Name for the deployment
            min_replica_count (int): Minimum number of replicas
            max_replica_count (int): Maximum number of replicas

        Returns:
            aiplatform.MatchingEngineIndexEndpoint: Updated endpoint

        Raises:
            VectorStoreError: If deployment creation fails
        """
        try:
            # Get the index and endpoint
            index = self.get_index(index_id)
            endpoint = self.get_index_endpoint(endpoint_id)

            # Generate deployment ID if not provided
            deployed_index_name += str(ULID())

            # Deploy the index
            endpoint = endpoint.deploy_index(
                index=index,
                deployed_index_id=deployed_index_name,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )

            logger.info(
                f"Successfully deployed index {index_id} to endpoint {endpoint_id} "
                f"with deployment ID {deployed_index_name}"
            )
            return endpoint

        except Exception as e:
            logger.error(f"Failed to create index deployment: {str(e)}")
            raise VectorStoreError(f"Failed to create index deployment: {str(e)}")

    @retry_with_backoff()
    def get_index_deployment(
        self, deployed_index_id: str
    ) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Retrieves a specific deployment from an endpoint.

        Args:
            deployed_index_id (str): ID of the deployed index to retrieve

        Returns:
            aiplatform.MatchingEngineIndexEndpoint: Retrieved deployment

        Raises:
            VectorStoreError: If deployment retrieval fails
        """
        try:
            # List all endpoints to find the one containing the deployment
            endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
            for endpoint in endpoints:
                for deployed_index in endpoint.deployed_indexes:
                    if deployed_index.id == deployed_index_id:
                        logger.info(f"Retrieved deployment {deployed_index_id}")
                        return endpoint

            raise VectorStoreError(f"Deployment {deployed_index_id} not found")

        except Exception as e:
            logger.error(f"Failed to get deployment: {str(e)}")
            raise VectorStoreError(f"Failed to get deployment: {str(e)}")

    @retry_with_backoff()
    def delete_index_deployment(
        self, endpoint_id: str, deployed_index_id: str
    ) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Deletes a deployment from an endpoint.

        Args:
            endpoint_id (str): ID of the endpoint containing the deployment
            deployed_index_id (str): ID of the deployment to delete

        Returns:
            aiplatform.MatchingEngineIndexEndpoint: Updated endpoint

        Raises:
            VectorStoreError: If deployment deletion fails
        """
        try:
            # Get the endpoint
            endpoint = self.get_index_endpoint(endpoint_id)

            # Undeploy the index
            endpoint = endpoint.undeploy_index(deployed_index_id=deployed_index_id)

            logger.info(
                f"Successfully deleted deployment {deployed_index_id} "
                f"from endpoint {endpoint_id}"
            )
            return endpoint

        except Exception as e:
            logger.error(f"Failed to delete index deployment: {str(e)}")
            raise VectorStoreError(f"Failed to delete index deployment: {str(e)}")

    @retry_with_backoff()
    def find_neighbors(
        self,
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
                client_options={"api_endpoint": self.api_endpoint}
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

            logging.info("Generated request to find neighbors: {request}")

            response = client.find_neighbors(request)
            logger.info(f"Successfully found {neighbor_count} neighbors")
            return response

        except Exception as e:
            logger.error(f"Failed to find neighbors: {str(e)}")
            raise VectorStoreError(f"Failed to find neighbors: {str(e)}")

    def list_deployments(self, endpoint_id: str) -> List[dict]:
        """
        Lists all deployments on an endpoint.

        Args:
            endpoint_id (str): ID of the endpoint to list deployments from

        Returns:
            List[dict]: List of deployment information

        Raises:
            VectorStoreError: If listing deployments fails
        """
        try:
            endpoint = self.get_index_endpoint(endpoint_id)
            deployments = [
                {
                    "deployed_index_id": d.deployed_index_id,
                    "index_resource_name": d.index,
                    "min_replica_count": d.min_replica_count,
                    "max_replica_count": d.max_replica_count,
                }
                for d in endpoint.deployed_indexes
            ]
            logger.info(
                f"Listed {len(deployments)} deployments on endpoint {endpoint_id}"
            )
            return deployments

        except Exception as e:
            logger.error(f"Failed to list deployments: {str(e)}")
            raise VectorStoreError(f"Failed to list deployments: {str(e)}")
