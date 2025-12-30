"""
OCR Pipeline - PDF text extraction using Mistral OCR.
Refactored from test_ocr.py
"""

import os
import tempfile
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCRPipeline:
    """
    OCR Pipeline for extracting text from PDF files using Mistral OCR API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OCR pipeline.
        
        Args:
            api_key: Mistral API key. If not provided, reads from MISTRAL_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._client = None
    
    @property
    def client(self):
        """Lazy load Mistral client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY not configured. Set it in environment variables.")
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a single PDF file using Mistral OCR API.
        
        Args:
            pdf_path: Full path to the PDF file
        
        Returns:
            Extracted text from the PDF
        """
        try:
            # Upload file to Mistral (FIXED: correct file upload format)
            filename = os.path.basename(pdf_path)
            with open(pdf_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": filename,
                        "content": f
                    },
                    purpose="ocr"
                )
            
            # Get signed URL
            file_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Run OCR
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": file_url.url
                },
                include_image_base64=False,
            )
            
            # Extract text from all pages
            text_parts = []
            for page in response.pages:
                page_text = getattr(page, "markdown", "") or getattr(page, "text", "")
                text_parts.append(page_text.strip())
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path}: {str(e)}")
            raise
    
    def process_pdf_to_file(self, pdf_path: str, output_dir: Path, base_dir: Optional[Path] = None, attempt: int = 1) -> bool:
        """
        Process a single PDF file and save result to a text file.
        
        Args:
            pdf_path: Full path to the PDF file
            output_dir: Directory where output .txt files will be saved
            base_dir: Base directory to compute relative paths (for preserving structure)
            attempt: Retry attempt number for logging
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Compute output .txt path mirroring directory structure
            if base_dir:
                relative_path = os.path.relpath(pdf_path, base_dir)
                txt_path = output_dir / (os.path.splitext(relative_path)[0] + ".txt")
            else:
                pdf_name = Path(pdf_path).stem
                txt_path = output_dir / f"{pdf_name}.txt"
            
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process PDF
            text = self.process_pdf(pdf_path)
            
            # Write output
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            logger.info(f"Successfully processed {pdf_path} -> {txt_path} (attempt {attempt})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path} (attempt {attempt}): {str(e)}")
            # Log error with context
            error_path = output_dir / "errors.txt"
            error_entry = f"[ERROR {pdf_path} attempt {attempt}: {str(e)}]\n"
            with open(error_path, "a", encoding="utf-8") as ef:
                ef.write(error_entry)
            return False
    
    def get_all_pdfs(self, folder: str) -> List[str]:
        """Get all PDF file paths recursively from a folder."""
        pdf_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
        return pdf_files
    
    def process_directory(self, directory: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process all PDF files in a directory recursively.
        
        Args:
            directory: Directory containing PDFs
            output_dir: Output directory for text files
        
        Returns:
            Dict with processing results
        """
        results = {"processed": 0, "failed": 0, "files": []}
        pdf_files = self.get_all_pdfs(str(directory))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return results
        
        logger.info(f"Processing {len(pdf_files)} PDFs from {directory}")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            if self.process_pdf_to_file(pdf_path, output_dir, base_dir=directory):
                results["processed"] += 1
                results["files"].append(os.path.basename(pdf_path))
            else:
                results["failed"] += 1
        
        return results
    
    def process_input(self, input_path: Path, output_dir: Path, return_text_for_single_pdf: bool = True) -> Dict[str, Any]:
        """
        Handle different input types: PDF file, ZIP file, or directory.
        
        Args:
            input_path: Path to input (PDF/ZIP/directory)
            output_dir: Output directory
            return_text_for_single_pdf: Whether to return extracted text for single PDF files
        
        Returns:
            Dict with processing results
        """
        results = {"processed": 0, "failed": 0, "files": [], "text": None}
        temp_dir = None
        
        try:
            if input_path.is_file():
                suffix = input_path.suffix.lower()
                if suffix == ".pdf":
                    # Single PDF file - FIXED: consistent behavior with file output + optional text return
                    output_dir.mkdir(parents=True, exist_ok=True)
                    success = self.process_pdf_to_file(str(input_path), output_dir)
                    
                    if success and return_text_for_single_pdf:
                        # Also return text for API compatibility
                        results["text"] = self.process_pdf(str(input_path))
                    
                    results["processed"] = 1 if success else 0
                    results["failed"] = 0 if success else 1
                    results["files"].append(input_path.name)
                    
                elif suffix == ".zip":
                    # ZIP file - extract and process PDFs
                    zip_stem = input_path.stem
                    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_zip_"))
                    extract_dir = temp_dir / zip_stem
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    
                    with zipfile.ZipFile(input_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    target_output = output_dir / zip_stem
                    target_output.mkdir(parents=True, exist_ok=True)
                    results = self.process_directory(extract_dir, target_output)
                    
            elif input_path.is_dir():
                # Directory containing PDFs
                output_dir.mkdir(parents=True, exist_ok=True)
                results = self.process_directory(input_path, output_dir)
                
        except Exception as e:
            logger.error(f"Failed to process input {input_path}: {str(e)}")
            results["failed"] += 1
            
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return results


# Module-level convenience function
def process_pdf(pdf_path: str, api_key: Optional[str] = None) -> str:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Optional Mistral API key
    
    Returns:
        Extracted text
    """
    pipeline = OCRPipeline(api_key=api_key)
    return pipeline.process_pdf(pdf_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR processing for PDF files and ZIP archives.")
    parser.add_argument("--input", required=True, help="Path to .pdf/.zip file or folder containing PDFs.")
    parser.add_argument("--output", required=True, help="Directory to store OCR output .txt files.")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        exit(1)
    
    pipeline = OCRPipeline()
    results = pipeline.process_input(input_path, output_dir)
    
    print(f"\n🎉 OCR Complete!")
    print(f"✅ Processed: {results['processed']}, Failed: {results['failed']}")
    if results.get("files"):
        print(f"📄 Files: {', '.join(results['files'][:5])}" + 
              (f" (and {len(results['files'])-5} more)" if len(results['files']) > 5 else ""))
