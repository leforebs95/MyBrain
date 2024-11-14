from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import io
import base64
from pathlib import Path
import json
from typing import Dict, Optional, List
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

# Import existing components
from claude_client import RAGClaudeClient
from embedding_generator import EmbeddingGenerator
from storage_manager import GCPStorageManager
from pdf_reader import PDFReader
from brain_processor import BrainProcessor, ProcessingJob
from database_manager import DatabaseConfig
from vector_store_manager import VectorStoreConfig, VectorStoreManager

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size


# Initialize components
def init_processors(config: Dict) -> tuple:
    """Initialize all necessary processors and managers"""
    storage_manager = GCPStorageManager(project_id=config["gcp_project_id"])
    pdf_reader = PDFReader(storage_manager)
    brain_processor = BrainProcessor(config, max_workers=4)
    vector_store = VectorStoreManager(config["vector_store"])
    embedding_generator = EmbeddingGenerator(config["voyage_api_key"])
    rag_claude = RAGClaudeClient(
        config["anthropic_api_key"],
        "my_brain_example_deployment_01JCMN355HEGT61CDQFBZYPK7R",  # You might want to make this configurable
        vector_store,
        embedding_generator,
        brain_processor.database_manager,
    )
    return storage_manager, pdf_reader, brain_processor, rag_claude


# Load configuration (you'll need to implement this based on your setup)
def load_config():
    """Load configuration from environment or config file"""
    return {
        "gcp_project_id": "mybrain-438918",
        "database": DatabaseConfig(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME"),
            instance_connection_name="mybrain-438918:us-central1:mybrain-oltp-db",
        ),
        "vector_store": VectorStoreConfig(
            project_id="mybrain-438918",
            location="us-central1",
            api_endpoint=os.getenv("MY_BRAIN_API_ENDPOINT"),
        ),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "voyage_api_key": os.getenv("VOYAGE_API_KEY"),
    }


# Initialize components
config = load_config()
storage_manager, pdf_reader, brain_processor, rag_claude = init_processors(config)

# Ensure upload directory exists
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pdf_previews(pdf_path: str) -> List[str]:
    """Convert all pages of PDF to base64 encoded images"""
    try:
        # Convert all pages
        images = convert_from_path(pdf_path)
        previews = []

        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            previews.append(base64.b64encode(img_byte_arr).decode())

        return previews
    except Exception as e:
        print(f"Error generating PDF previews: {str(e)}")
        return []


def process_pdf(local_path: str, filename: str) -> Dict:
    """Process PDF through the brain processing pipeline"""
    try:
        # Upload to GCP and process
        output_prefix = f"handwritten-notes/{filename.replace('.pdf', '')}"
        gcp_paths = pdf_reader.process_local_pdf(
            local_path,
            output_bucket="my-brain-vector-store",
            output_prefix=output_prefix,
        )

        # Create and process jobs - one for each page
        jobs = []
        for i, path in enumerate(gcp_paths):
            job = ProcessingJob(
                input_bucket="my-brain-vector-store",
                input_pdf=path,
                output_bucket="my-brain-vector-store",
                output_base=f"handwritten-ocr/{filename.replace('.pdf', '_')}",
            )
            jobs.append(job)

        # Process all pages
        results, progress = brain_processor.batch_process_documents(jobs)

        if results:
            # Format results by page
            pages = []
            for i, result in enumerate(results):
                pages.append(
                    {
                        "page_number": i + 1,
                        "id": result.id,
                        "input_pdf": result.input_pdf,
                        "output_base": result.output_base,
                        "original_ocr": result.original_ocr,
                        "improved_ocr": result.improved_ocr,
                        "embedding": result.embedding,
                        "metadata": (
                            result.metadata if hasattr(result, "metadata") else {}
                        ),
                    }
                )

            return {"success": True, "pages": pages}
        else:
            return {"success": False, "error": "No results returned from processing"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle PDF file upload, processing, and return results"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Generate previews for all pages
        previews = get_pdf_previews(filepath)

        # Process the PDF
        processing_results = process_pdf(filepath, filename)

        if previews and processing_results["success"]:
            return jsonify(
                {
                    "message": "File processed successfully",
                    "filename": filename,
                    "previews": previews,
                    "pages": processing_results["pages"],
                }
            )
        else:
            error_message = processing_results.get("error", "Failed to process file")
            return jsonify({"error": error_message}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save_embeddings", methods=["POST"])
def save_embeddings():
    """Save embeddings using brain processor"""
    try:
        data = request.json
        if not data or "pages" not in data:
            return jsonify({"error": "No results provided"}), 400

        # Convert the JSON results back to OCRResult objects
        ocr_results = []
        for page in data["pages"]:
            ocr_result = OCRResult(
                id=page["id"],
                input_pdf=page["input_pdf"],
                output_base=page["output_base"],
                original_ocr=page.get("original_ocr"),
                improved_ocr=page.get("improved_ocr"),
                embedding=page.get("embedding"),
                metadata=page.get("metadata", {}),
            )
            ocr_results.append(ocr_result)

        # Store the embeddings
        success, progress = brain_processor.store_embeddings(ocr_results)

        if success:
            return jsonify(
                {
                    "message": "Embeddings saved successfully",
                    "progress": progress.to_dict() if progress else None,
                }
            )
        else:
            return jsonify({"error": "Failed to save embeddings"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    """Render the upload form"""
    return render_template("index.html")


@app.route("/chat")
def chat_interface():
    """Render the chat interface"""
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages and return responses"""
    try:
        data = request.json
        query = data.get("message")

        if not query:
            return jsonify({"error": "No message provided"}), 400

        # Get response from RAG Claude
        response = rag_claude.chat(query)

        return jsonify({"response": response, "success": True})

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    """Clear the chat conversation history"""
    try:
        rag_claude.clear_conversation()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(debug=True)
