from typing import List
import voyageai
import logging


from errors import EmbeddingError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using Voyage AI API.

    Attributes:
        client (voyageai.Client): Voyage AI client
    """

    def __init__(self, voyage_api_key: str):
        """
        Initialize the embedding generator.

        Args:
            voyage_api_key (str): Voyage AI API key
        """
        try:
            self.client = voyageai.Client(api_key=voyage_api_key)
            logger.info("Initialized Voyage AI client")
        except Exception as e:
            logger.error(f"Failed to initialize Voyage AI client: {str(e)}")
            raise EmbeddingError(f"Voyage AI client initialization failed: {str(e)}")

    @retry_with_backoff()
    def generate_embeddings(
        self, texts: List[str], model: str = "voyage-3", input_type: str = "document"
    ) -> List[List[float]]:
        """
        Generates embeddings for given texts.

        Args:
            texts (List[str]): Texts to generate embeddings for
            model (str): Model to use for embedding generation
            input_type (str): Type of input text

        Returns:
            List[List[float]]: Generated embeddings

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if input_type not in ["query", "document"]:
            logger.error(f"Invalid input_type: {input_type}")
            raise ValueError(
                f"Invalid input_type: {input_type}. Must be 'query' or 'document'."
            )
        try:
            result = self.client.embed(texts, model=model, input_type=input_type)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return result.embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
