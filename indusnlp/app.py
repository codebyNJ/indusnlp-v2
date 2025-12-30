"""
Flask API for IndusNLP - OCR, Text Cleaning, and Q&A Generation endpoints.
Uses the pipeline modules from indusnlp.pipelines for core processing logic.
"""

import os
import sys
import tempfile
import shutil
import zipfile
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indusnlp.pipelines.cleaning import CleaningPipeline
from indusnlp.pipelines.ocr import OCRPipeline
from indusnlp.pipelines.qna import QnAPipeline

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

# ============================================================
# Global Configuration
# ============================================================
UPLOAD_FOLDER = tempfile.mkdtemp(prefix="indusnlp_uploads_")
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="indusnlp_outputs_")

# ============================================================
# Utility Functions
# ============================================================
def _to_bool(v, default):
    """Normalize value to boolean with fallback."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    if v is None:
        return default
    return bool(v)

def _parse_int(value, default, min_val, max_val):
    """Parse integer with bounds checking and fallback."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_val, min(max_val, v))

# ============================================================
# Error Handler Decorator (Fixed)
# ============================================================
def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except HTTPException as e:
            # ✅ Preserve HTTPException status codes
            return jsonify({"success": False, "error": e.description}), e.code
        except Exception as e:
            # Fallback: internal server error
            return jsonify({"success": False, "error": str(e)}), 500
    return decorated_function

# ============================================================
# Temp Directory Cleanup
# ============================================================
@app.teardown_appcontext
def cleanup_temp_dirs(exception=None):
    """Clean up global temp directories on app context teardown."""
    for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

# ============================================================
# Health Check Endpoint
# ============================================================
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "IndusNLP API",
        "version": "2.0.1",  # Updated version
        "endpoints": {
            "ocr": "/api/ocr",
            "clean": "/api/clean",
            "qna": "/api/qna"
        }
    })

# ============================================================
# OCR Endpoint
# ============================================================
@app.route("/api/ocr", methods=["POST"])
@handle_errors
def ocr_endpoint():
    """
    OCR endpoint - Process PDF files using Mistral OCR.
    
    Accepts:
    - Single PDF file upload
    - ZIP file containing PDFs
    
    Returns:
    - JSON with extracted text for single file
    - ZIP file with text files for multiple PDFs
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return jsonify({
            "success": False,
            "error": "MISTRAL_API_KEY not configured. Set it in environment variables."
        }), 400
    
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".zip")):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Only PDF and ZIP files are supported."
        }), 400
    
    # Initialize OCR pipeline
    ocr_pipeline = OCRPipeline(api_key=api_key)
    
    # Save uploaded file
    temp_input = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_input)
    
    try:
        input_path = Path(temp_input)
        output_dir = Path(OUTPUT_FOLDER)
        
        if filename.endswith(".pdf"):
            # Single PDF processing using pipeline
            text = ocr_pipeline.process_pdf(temp_input)
            return jsonify({
                "success": True,
                "filename": file.filename,
                "text": text
            })
        else:
            # ZIP file processing using pipeline
            results = ocr_pipeline.process_input(input_path, output_dir)
            
            # Create output ZIP from results
            output_zip = os.path.join(OUTPUT_FOLDER, "ocr_results.zip")
            with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        if f.endswith(".txt"):
                            file_path = os.path.join(root, f)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
            
            return send_file(
                output_zip,
                mimetype="application/zip",
                as_attachment=True,
                download_name="ocr_results.zip"
            )
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

# ============================================================
# Text Cleaning Endpoint (Fixed)
# ============================================================
@app.route("/api/clean", methods=["POST"])
@handle_errors
def clean_endpoint():
    """
    Text cleaning endpoint using CleaningPipeline.
    
    Accepts:
    - JSON body with "text" field
    - Single TXT file upload
    - ZIP file containing TXT files
    
    Options (query params or JSON):
    - transliterate: bool (default: true) - Enable English to Hindi transliteration
    - filter_badwords: bool (default: true) - Filter bad words
    
    Returns:
    - JSON with cleaned text for single text/file
    - ZIP file with cleaned files for ZIP input
    """
    # Parse options from query params with defaults
    transliterate = request.args.get("transliterate", "true").lower() == "true"
    filter_badwords = request.args.get("filter_badwords", "true").lower() == "true"
    
    # Check if JSON body
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "")
        
        # ✅ Fixed: Normalize booleans from JSON with proper fallback
        transliterate = _to_bool(data.get("transliterate"), transliterate)
        filter_badwords = _to_bool(data.get("filter_badwords"), filter_badwords)
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Use CleaningPipeline
        pipeline = CleaningPipeline(transliterate=transliterate, filter_badwords=filter_badwords)
        cleaned = pipeline.clean_text(text)
        
        return jsonify({
            "success": True,
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "text": cleaned
        })
    
    # File upload
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file or text provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    filename = file.filename.lower()
    if not (filename.endswith(".txt") or filename.endswith(".zip")):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Only TXT and ZIP files are supported."
        }), 400
    
    # Initialize pipeline
    pipeline = CleaningPipeline(transliterate=transliterate, filter_badwords=filter_badwords)
    
    temp_input = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_input)
    
    try:
        if filename.endswith(".txt"):
            # Single file processing
            with open(temp_input, "r", encoding="utf-8") as f:
                text = f.read()
            
            cleaned = pipeline.clean_text(text)
            
            return jsonify({
                "success": True,
                "filename": file.filename,
                "original_length": len(text),
                "cleaned_length": len(cleaned),
                "text": cleaned
            })
        else:
            # ZIP file processing
            input_path = Path(temp_input)
            output_dir = Path(tempfile.mkdtemp(prefix="clean_output_"))
            
            try:
                results = pipeline.process_files(input_path, output_dir)
                
                # Create output ZIP
                output_zip = os.path.join(OUTPUT_FOLDER, "cleaned_results.zip")
                with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(output_dir):
                        for f in files:
                            file_path = os.path.join(root, f)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
                
                return send_file(
                    output_zip,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name="cleaned_results.zip"
                )
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

# ============================================================
# Q&A Generation Endpoint (Fixed)
# ============================================================
@app.route("/api/qna", methods=["POST"])
@handle_errors
def qna_endpoint():
    """
    Q&A Generation endpoint using QnAPipeline.
    
    Accepts:
    - JSON body with "text" field
    - Single TXT file upload
    - ZIP file containing TXT files
    
    Options (query params or JSON):
    - num_questions: int (default: 25, 1-300) - Number of questions to generate
    - batch_size: int (default: 25, 1-50) - Questions per API call
    
    Returns:
    - JSON with generated Q&A pairs
    - ZIP file with Q&A files for ZIP input
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "success": False,
            "error": "GEMINI_API_KEY not configured. Set it in environment variables."
        }), 400
    
    # ✅ Fixed: Parse integers with bounds validation
    num_questions = _parse_int(request.args.get("num_questions"), 25, 1, 300)
    batch_size = _parse_int(request.args.get("batch_size"), 25, 1, 50)
    
    # Initialize Q&A pipeline
    qna_pipeline = QnAPipeline(api_key=api_key)
    
    # Check if JSON body
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "")
        
        # ✅ Fixed: Parse integers from JSON with bounds validation
        num_questions = _parse_int(data.get("num_questions"), num_questions, 1, 300)
        batch_size = _parse_int(data.get("batch_size"), batch_size, 1, 50)
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        qna_list = qna_pipeline.generate_qna(text, num_questions, batch_size)
        
        return jsonify({
            "success": True,
            "num_questions": len(qna_list),
            "qna": qna_list
        })
    
    # File upload
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file or text provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    filename = file.filename.lower()
    if not (filename.endswith(".txt") or filename.endswith(".zip")):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Only TXT and ZIP files are supported."
        }), 400
    
    temp_input = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_input)
    
    try:
        if filename.endswith(".txt"):
            # Single file processing
            with open(temp_input, "r", encoding="utf-8") as f:
                text = f.read()
            
            qna_list = qna_pipeline.generate_qna(text, num_questions, batch_size)
            
            return jsonify({
                "success": True,
                "filename": file.filename,
                "num_questions": len(qna_list),
                "qna": qna_list
            })
        else:
            # ZIP file processing
            input_path = Path(temp_input)
            output_dir = Path(tempfile.mkdtemp(prefix="qna_output_"))
            
            try:
                results = qna_pipeline.process_input(input_path, output_dir, num_questions, batch_size)
                
                # Create output ZIP
                output_zip = os.path.join(OUTPUT_FOLDER, "qna_results.zip")
                with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(output_dir):
                        for f in files:
                            file_path = os.path.join(root, f)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
                
                return send_file(
                    output_zip,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name="qna_results.zip"
                )
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

# ============================================================
# Main Entry Point
# ============================================================
def main():
    """Run the Flask development server."""
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

if __name__ == "__main__":
    main()
