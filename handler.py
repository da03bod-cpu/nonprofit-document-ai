import os
import uuid
import requests
import runpod
from ocr import process_document

TMP_DIR = "/tmp/documents"
os.makedirs(TMP_DIR, exist_ok=True)

def download_file(url: str, output_path: str):
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def handler(job):
    job_input = job.get("input", {})
    file_url = job_input.get("file_url")
    if not file_url:
        raise ValueError("Missing input.file_url")

    job_id = str(uuid.uuid4())
    file_path = os.path.join(TMP_DIR, f"{job_id}.pdf")

    try:
        print(f"Downloading document: {file_url}")
        download_file(file_url, file_path)
        print("Document downloaded.")
        print("Starting PaddleOCR-VL 1.6...")
        result = process_document(file_path)
        print("OCR completed.")
        return {"status": "success", "document_id": job_id, "result": result}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

runpod.serverless.start({"handler": handler})
