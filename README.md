# Nonprofit Document AI

Initial RunPod Serverless worker for PaddleOCR-VL 1.6.

Flow:
PDF URL -> RunPod Worker -> PaddleOCR-VL 1.6 -> OCR JSON

Qwen3 will be added after OCR validation.

## Input
```json
{"input":{"file_url":"https://example.com/report.pdf"}}
```

Recommended first GPU: NVIDIA L4 24GB.
Fallback: A10G 24GB.
