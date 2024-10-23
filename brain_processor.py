from typing import List, Dict, Optional
import json
import logging
from dataclasses import dataclass
from functools import wraps
from time import sleep

from storage_manager import BrainStorageManager
from ocr_processor import OCRProcessor
from text_improver import TextImprover
from embedding_generator import EmbeddingGenerator
from vector_store_manager import VectorStoreManager
from errors import BrainProcessingError, StorageError


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """
    Data class to store OCR processing results.

    Attributes:
        input_pdf (str): Path to the input PDF file
        output_json (str): Path to the output JSON file
        original_ocr (str, optional): Raw OCR text
        improved_ocr (str, optional): Improved OCR text
        embedding (List[float], optional): Vector embedding of the improved text
    """

    input_pdf: str
    output_json: str
    original_ocr: Optional[str] = None
    improved_ocr: Optional[str] = None
    embedding: Optional[List[float]] = None


class BrainProcessor:
    """
    Main class that orchestrates the entire brain processing pipeline.

    Attributes:
        storage_manager (BrainStorageManager): Storage manager instance
        ocr_processor (OCRProcessor): OCR processor instance
        text_improver (TextImprover): Text improver instance
        embedding_generator (EmbeddingGenerator): Embedding generator instance
        vector_store (VectorStoreManager): Vector store manager instance
    """

    def __init__(self, config: Dict[str, str]):
        """
        Initialize the brain processor.

        Args:
            config (Dict[str, str]): Configuration dictionary containing necessary API keys and settings
        """
        try:
            self.storage_manager = BrainStorageManager(
                project_id=config["project_id"],
                brain_bucket=config["brain_bucket"],
                vs_bucket=config["vs_bucket"],
            )
            self.ocr_processor = OCRProcessor(self.storage_manager)
            self.text_improver = TextImprover(config["anthropic_api_key"])
            self.embedding_generator = EmbeddingGenerator(config["voyage_api_key"])
            self.vector_store = VectorStoreManager(config["project_id"])
            logger.info("Initialized BrainProcessor with all components")
        except Exception as e:
            logger.error(f"Failed to initialize BrainProcessor: {str(e)}")
            raise BrainProcessingError(
                f"BrainProcessor initialization failed: {str(e)}"
            )

    def process_document(self, input_pdf: str) -> OCRResult:
        """
        Processes a single document through the entire pipeline.

        Args:
            input_pdf (str): Path to the input PDF file

        Returns:
            OCRResult: Processing results

        Raises:
            BrainProcessingError: If any step in the pipeline fails
        """
        try:
            logger.info(f"Starting processing for document: {input_pdf}")

            # Construct paths
            gcs_source_uri = f"gs://{self.storage_manager.brain_bucket}/{input_pdf}"
            output_base = input_pdf.replace(".pdf", "_")
            gcs_destination_uri = f"gs://{self.storage_manager.vs_bucket}/{output_base}"

            # Create result object
            result = OCRResult(
                input_pdf=input_pdf, output_json=f"{output_base}output-1-to-1.json"
            )

            # Process document through pipeline
            self.ocr_processor.process_document(gcs_source_uri, gcs_destination_uri)
            result.original_ocr = self.ocr_processor.extract_text(result.output_json)
            result.improved_ocr = self.text_improver.improve_text(result.original_ocr)
            result.embedding = self.embedding_generator.generate_embeddings(
                [result.improved_ocr]
            )[0]

            logger.info(f"Successfully completed processing for document: {input_pdf}")
            return result

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise BrainProcessingError(f"Document processing failed: {str(e)}")

    def batch_process_documents(self, input_pdfs: List[str]) -> List[OCRResult]:
        """
        Processes multiple documents through the pipeline.

        Args:
            input_pdfs (List[str]): List of paths to input PDF files

        Returns:
            List[OCRResult]: List of processing results

        Raises:
            BrainProcessingError: If processing fails for any document
        """
        results = []
        failed_documents = []

        for input_pdf in input_pdfs:
            try:
                result = self.process_document(input_pdf)
                results.append(result)
                logger.info(f"Successfully processed document: {input_pdf}")
            except Exception as e:
                logger.error(f"Failed to process document {input_pdf}: {str(e)}")
                failed_documents.append((input_pdf, str(e)))

        if failed_documents:
            failure_messages = "\n".join(
                [f"{doc}: {err}" for doc, err in failed_documents]
            )
            logger.error(
                f"Batch processing completed with failures:\n{failure_messages}"
            )
            raise BrainProcessingError(
                f"Batch processing failed for some documents:\n{failure_messages}"
            )

        return results

    def save_results(self, results: List[OCRResult], output_prefix: str) -> None:
        """
        Saves processing results to storage.

        Args:
            results (List[OCRResult]): Processing results to save
            output_prefix (str): Prefix for output files

        Raises:
            StorageError: If saving results fails
        """
        try:
            for i, result in enumerate(results):
                output_file = f"{output_prefix}_{i}.json"
                result_dict = {
                    "input_pdf": result.input_pdf,
                    "output_json": result.output_json,
                    "original_ocr": result.original_ocr,
                    "improved_ocr": result.improved_ocr,
                    "embedding": result.embedding,
                }

                self.storage_manager.upload_json(
                    json.dumps(result_dict), output_file, self.storage_manager.vs_bucket
                )
                logger.info(f"Saved results for {result.input_pdf} to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise StorageError(f"Failed to save processing results: {str(e)}")
