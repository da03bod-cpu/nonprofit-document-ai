# Nonprofit Document AI — PaddleOCR-VL 1.6 + Qwen3

Single RunPod Serverless endpoint.

## Routing

- PDF → PaddleOCR-VL 1.6 → Qwen3
- DOCX → python-docx → Qwen3
- JSON → JSON parser → Qwen3

## RunPod input

```json
{
  "input": {
    "file_url": "https://example.com/report.pdf",
    "file_type": "pdf"
  }
}
```

`file_type` can be `pdf`, `docx`, or `json`.

## Qwen model

Default:

`Qwen/Qwen3-8B`

Environment variables:

- `QWEN_MODEL`
- `QWEN_LOAD_IN_4BIT=true`
- `QWEN_MAX_NEW_TOKENS=4096`
- `QWEN_MAX_INPUT_CHARS=100000`

For a fine-tuned Qwen3 model, set `QWEN_MODEL` to the Hugging Face model/repository path available to the worker.

## Important

The endpoint now returns the final structured JSON from Qwen3. n8n does not need a separate Clean OCR node.

The current implementation truncates extremely large text at `QWEN_MAX_INPUT_CHARS`. For very large annual reports, the next production improvement should be page-aware chunking + Qwen merge rather than raw truncation.
