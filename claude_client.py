import logging
from typing import Dict, List, Optional, Union, Any
import anthropic
from dataclasses import dataclass

from errors import BrainProcessingError
from utils import retry_with_backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClaudeMessage:
    """
    Represents a message in the Claude conversation.

    Attributes:
        role (str): Role of the message sender ('user' or 'assistant')
        content (Union[str, List[Dict[str, str]]]): Content of the message
    """

    role: str
    content: Union[str, List[Dict[str, str]]]


@dataclass
class ClaudeResponse:
    """
    Represents a response from Claude.

    Attributes:
        content (str): The processed content from Claude
        raw_response (Any): The raw response object from the Anthropic API
    """

    content: str
    raw_response: Any


class ClaudeClient:
    """
    A generic interface for interacting with Claude AI.

    This client provides methods for various text processing tasks using the Claude API.
    It handles message construction, error handling, and response processing.

    Attributes:
        client (anthropic.Anthropic): The Anthropic API client
        default_model (str): Default Claude model to use
        default_max_tokens (int): Default maximum tokens for responses
        default_temperature (float): Default temperature for response generation
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0

    def __init__(
        self,
        api_key: str,
        default_model: Optional[str] = None,
        default_max_tokens: Optional[int] = None,
        default_temperature: Optional[float] = None,
    ):
        """
        Initialize the Claude client.

        Args:
            api_key (str): Anthropic API key
            default_model (str, optional): Default model to use
            default_max_tokens (int, optional): Default maximum tokens
            default_temperature (float, optional): Default temperature

        Raises:
            BrainProcessingError: If client initialization fails
        """
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.default_model = default_model or self.DEFAULT_MODEL
            self.default_max_tokens = default_max_tokens or self.DEFAULT_MAX_TOKENS
            self.default_temperature = default_temperature or self.DEFAULT_TEMPERATURE
            logger.info("Initialized Claude client")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {str(e)}")
            raise BrainProcessingError(f"Claude client initialization failed: {str(e)}")

    @retry_with_backoff()
    def process_text(
        self,
        text: str,
        system_message: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ClaudeResponse:
        """
        Process text using Claude with a specific system message.

        Args:
            text (str): Text to process
            system_message (str): System message defining the task
            model (str, optional): Model to use
            max_tokens (int, optional): Maximum tokens in response
            temperature (float, optional): Temperature for response generation
            **kwargs: Additional arguments to pass to the API

        Returns:
            ClaudeResponse: Processed response from Claude

        Raises:
            BrainProcessingError: If text processing fails
        """
        try:
            message = self.client.messages.create(
                model=model or self.default_model,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": text}]}
                ],
                **kwargs,
            )

            content = "\n".join([block.text for block in message.content])
            logger.info("Successfully processed text with Claude")
            return ClaudeResponse(content=content, raw_response=message)

        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            raise BrainProcessingError(f"Claude text processing failed: {str(e)}")

    @retry_with_backoff()
    def chat(
        self,
        messages: List[ClaudeMessage],
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ClaudeResponse:
        """
        Have a chat conversation with Claude.

        Args:
            messages (List[ClaudeMessage]): List of messages in the conversation
            system_message (str, optional): System message for the conversation
            model (str, optional): Model to use
            max_tokens (int, optional): Maximum tokens in response
            temperature (float, optional): Temperature for response generation
            **kwargs: Additional arguments to pass to the API

        Returns:
            ClaudeResponse: Response from Claude

        Raises:
            BrainProcessingError: If chat processing fails
        """
        try:
            formatted_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            message = self.client.messages.create(
                model=model or self.default_model,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                system=system_message,
                messages=formatted_messages,
                **kwargs,
            )

            content = "\n".join([block.text for block in message.content])
            logger.info("Successfully processed chat with Claude")
            return ClaudeResponse(content=content, raw_response=message)

        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            raise BrainProcessingError(f"Claude chat processing failed: {str(e)}")


# Example implementation of the TextImprover using the new ClaudeClient
class TextImprover:
    """
    Improves OCR text using the Claude client.

    This class serves as an example of how to use the ClaudeClient
    for a specific text processing task.
    """

    SYSTEM_MESSAGE = """
    You will be receiving text blobs extracted from PDF scans of handwritten notes.
    Your task is to improve formatting, fix indentation, and ensure proper line spacing
    while maintaining the original content and meaning.
    """

    def __init__(self, api_key: str):
        """
        Initialize the text improver.

        Args:
            api_key (str): Anthropic API key
        """
        self.claude_client = ClaudeClient(api_key)

    def improve_text(self, raw_text: str) -> str:
        """
        Improves OCR text formatting.

        Args:
            raw_text (str): Raw OCR text to improve

        Returns:
            str: Improved text

        Raises:
            BrainProcessingError: If the text improvement fails
        """
        try:
            response = self.claude_client.process_text(
                text=raw_text, system_message=self.SYSTEM_MESSAGE
            )
            return response.content
        except BrainProcessingError as e:
            logger.error(f"Text improvement failed: {str(e)}")
            raise BrainProcessingError(f"Text improvement failed: {str(e)}")
