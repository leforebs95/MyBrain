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

    SYSTEM_MESSAGE = "Please improve the following text."

    def __init__(self, api_key: str):
        """
        Initialize the text improver.

        Args:
            api_key (str): Anthropic API key
        """
        self.claude_client = ClaudeClient(api_key)
        with open("OCRImprovementSystemPrompt.txt", "r") as file:
            self.SYSTEM_MESSAGE = file.read()

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


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str
    content: str
    timestamp: datetime = None


@dataclass
class ContextWindow:
    """Represents the context window for RAG."""

    text_chunks: List[str]
    total_tokens: int
    max_tokens: int = 4000  # Default max tokens for context

    def has_space(self, new_chunk_tokens: int) -> bool:
        """Check if new chunk can fit in context window."""
        return (self.total_tokens + new_chunk_tokens) <= self.max_tokens

    def add_chunk(self, chunk: str, chunk_tokens: int):
        """Add a new chunk to context window if space available."""
        if self.has_space(chunk_tokens):
            self.text_chunks.append(chunk)
            self.total_tokens += chunk_tokens
            return True
        return False


class ConversationManager:
    """Manages conversation history and context."""

    def __init__(self, max_history: int = 10):
        self.history: List[Message] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        message = Message(role=role, content=content, timestamp=datetime.now())
        self.history.append(message)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Get history formatted for Claude API."""
        return [{"role": msg.role, "content": msg.content} for msg in self.history]


class RAGClaudeClient:
    """
    Enhanced Claude client with RAG support.

    This client manages conversations, handles context windows,
    and provides RAG-specific functionality.
    """

    DEFAULT_SYSTEM_MESSAGE = "Please provide an answer to the following question."

    def __init__(
        self,
        api_key: str,
        vector_store,  # Reference to vector store for similarity search
        default_model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1000,
        temperature: float = 0,
    ):
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.vector_store = vector_store
            self.default_model = default_model
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.conversation = ConversationManager()
            self.context_window = ContextWindow([], 0)
            with open("ConversationSystemPrompt.txt", "r") as file:
                self.DEFAULT_SYSTEM_MESSAGE = file.read()
            logger.info("Initialized RAG Claude client")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Claude client: {str(e)}")
            raise BrainProcessingError(
                f"RAG Claude client initialization failed: {str(e)}"
            )

    def _get_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant context chunks based on query similarity.

        Args:
            query: User's question
            k: Number of similar chunks to retrieve

        Returns:
            List of relevant text chunks
        """
        try:
            results = self.vector_store.find_similar_documents(query, limit=k)
            return [r["improved_ocr"] for r in results]
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}")
            raise BrainProcessingError(f"Context retrieval failed: {str(e)}")

    def _build_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Build a prompt combining query and context.

        Args:
            query: User's question
            context_chunks: Retrieved relevant context

        Returns:
            Formatted prompt string
        """
        context_str = "\n\n".join(context_chunks)
        return f"""Context information is below.
---------------------
{context_str}
---------------------

Given the context information, please answer the following question:
{query}

Remember to only use information from the provided context. If the context doesn't 
contain enough information to fully answer the question, please say so."""

    @retry_with_backoff()
    def chat(
        self,
        query: str,
        system_message: Optional[str] = None,
        refresh_context: bool = True,
    ) -> str:
        """
        Process a chat message with RAG support.

        Args:
            query: User's question
            system_message: Optional custom system message
            refresh_context: Whether to fetch new context or use existing

        Returns:
            Assistant's response
        """
        try:
            # Get fresh context if requested
            if refresh_context:
                context_chunks = self._get_relevant_context(query)
                prompt = self._build_prompt(query, context_chunks)
            else:
                prompt = query

            # Add user message to conversation
            self.conversation.add_message("user", prompt)

            # Get response from Claude
            message = self.client.messages.create(
                model=self.default_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message or self.DEFAULT_SYSTEM_MESSAGE,
                messages=self.conversation.get_formatted_history(),
            )

            response = "\n".join([block.text for block in message.content])

            # Add assistant response to conversation
            self.conversation.add_message("assistant", response)

            return response

        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            raise BrainProcessingError(f"Chat processing failed: {str(e)}")

    def clear_conversation(self):
        """Reset the conversation history."""
        self.conversation = ConversationManager()
        logger.info("Cleared conversation history")
