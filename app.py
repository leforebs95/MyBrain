from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import io
import base64
from pathlib import Path
import json
from typing import Dict, Optional
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
        "test_data_example_deployment_01JCHA9B83RN0PAF3NHVBPVA9M",  # You might want to make this configurable
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


def get_pdf_preview(pdf_path: str) -> Optional[str]:
    """Convert first page of PDF to base64 encoded image"""
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if images:
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode()
    except Exception as e:
        print(f"Error generating PDF preview: {str(e)}")
        return None


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

        # Create and process job
        jobs = []
        for path in gcp_paths:
            jobs.append(
                ProcessingJob(
                    input_bucket="my-brain-vector-store",
                    input_pdf=path,
                    output_bucket="my-brain-vector-store",
                    output_base=f"handwritten-ocr/{filename.replace('.pdf', '_')}",
                )
            )

        # Process single document
        results, progress = brain_processor.batch_process_documents(jobs)

        if results and len(results) > 0:
            return {
                "original_ocr": results[0].original_ocr,
                "improved_ocr": results[0].improved_ocr,
                "embedding": results[0].embedding,
                "success": True,
            }
        else:
            return {"success": False, "error": "No results returned from processing"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.route("/")
def index():
    """Render the upload form"""
    return render_template("index.html")


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

        # Generate preview
        preview_base64 = get_pdf_preview(filepath)

        # Process the PDF
        processing_results = process_pdf(filepath, filename)

        if preview_base64 and processing_results["success"]:
            return jsonify(
                {
                    "message": "File processed successfully",
                    "filename": filename,
                    "preview": preview_base64,
                    "original_ocr": processing_results["original_ocr"],
                    "improved_ocr": processing_results["improved_ocr"],
                    "embedding": processing_results["embedding"],
                }
            )
        else:
            error_message = processing_results.get("error", "Failed to process file")
            return jsonify({"error": error_message}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
