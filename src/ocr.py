from pathlib import Path
from paddleocr import PaddleOCRVL

OUTPUT_DIR = Path("/tmp/ocr_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PaddleOCR-VL 1.6...")
pipeline = PaddleOCRVL()
print("PaddleOCR-VL 1.6 loaded.")

def process_document(file_path: str):
    results = pipeline.predict(file_path)
    pages = []
    for page_number, result in enumerate(results, start=1):
        pages.append({"page": page_number, "data": result.json})
    return {"model": "PaddleOCR-VL-1.6", "pages": pages}
