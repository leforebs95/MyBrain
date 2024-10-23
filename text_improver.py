import logging

import anthropic


from errors import BrainProcessingError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextImprover:
    """
    Improves OCR text using Claude API.

    Attributes:
        client (anthropic.Anthropic): Anthropic API client
        system_message (str): System message for Claude
    """

    def __init__(self, anthropic_api_key: str):
        """
        Initialize the text improver.

        Args:
            anthropic_api_key (str): Anthropic API key
        """
        try:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            logger.info("Initialized Anthropic API client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise BrainProcessingError(
                f"Anthropic client initialization failed: {str(e)}"
            )

        self.system_message = """
        You will be receiving text blobs extracted from PDF scans of handwritten notes.
        Your task is to improve formatting, fix indentation, and ensure proper line spacing
        while maintaining the original content and meaning.
        """

    @retry_with_backoff()
    def improve_text(
        self, raw_text: str, model: str = "claude-3-5-sonnet-20240620"
    ) -> str:
        """
        Improves OCR text formatting using Claude.

        Args:
            raw_text (str): Raw OCR text to improve
            model (str): Claude model to use

        Returns:
            str: Improved text

        Raises:
            BrainProcessingError: If the text improvement fails
        """
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0,
                system=self.system_message,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": raw_text}]}
                ],
            )
            improved_text = "\n".join(
                [text_block.text for text_block in message.content]
            )
            logger.info("Successfully improved text formatting")
            return improved_text
        except Exception as e:
            logger.error(f"Text improvement failed: {str(e)}")
            raise BrainProcessingError(f"Text improvement failed: {str(e)}")
