import os
import uuid
from pathlib import Path

import requests
import runpod

from ocr import process_pdf
from document_parser import parse_document
from qwen import QwenExtractor


TMP_DIR = "/tmp/documents"
os.makedirs(TMP_DIR, exist_ok=True)

# Load Qwen once when the worker starts.
# This avoids reloading the model for every request.
qwen = QwenExtractor()


def download_file(url: str, output_path: str):
    response = requests.get(
        url,
        stream=True,
        timeout=600,
        allow_redirects=True,
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def detect_extension(file_url: str, job_input: dict):
    explicit = job_input.get("file_type")

    if explicit:
        return explicit.lower().lstrip(".")

    clean_url = file_url.split("?", 1)[0]
    suffix = Path(clean_url).suffix.lower().lstrip(".")

    if suffix:
        return suffix

    raise ValueError(
        "Cannot detect file type. Provide input.file_type "
        "(pdf, docx, or json)."
    )


def build_source_text(file_path: str, extension: str):
    if extension == "pdf":
        ocr_result = process_pdf(file_path)

        return (
            ocr_result["full_text"],
            {
                "source_type": "pdf",
                "ocr": ocr_result,
            },
        )

    if extension in {"docx", "json"}:
        text = parse_document(file_path, extension)

        return (
            text,
            {
                "source_type": extension,
            },
        )

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported types: pdf, docx, json."
    )


def handler(job):
    job_input = job.get("input", {})

    file_url = job_input.get("file_url")
    if not file_url:
        raise ValueError("Missing input.file_url")

    extension = detect_extension(file_url, job_input)

    document_id = str(uuid.uuid4())
    file_path = os.path.join(
        TMP_DIR,
        f"{document_id}.{extension}",
    )

    try:
        print(f"Downloading {extension.upper()} document...")
        download_file(file_url, file_path)

        print("Extracting document content...")
        source_text, source_meta = build_source_text(
            file_path,
            extension,
        )

        print(
            f"Extracted {len(source_text)} characters. "
            "Sending to Qwen3..."
        )

        structured = qwen.generate(source_text)

        print("Qwen3 extraction completed.")

        return {
            "status": "success",
            "document_id": document_id,
            "source_type": extension,
            "result": structured,
            "metadata": {
                "model": qwen.model_name,
                "input_characters": len(source_text),
                "ocr_used": extension == "pdf",
            },
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


runpod.serverless.start({"handler": handler})
