from paddleocr import PaddleOCRVL

print("Loading PaddleOCR-VL 1.6...")
pipeline = PaddleOCRVL()
print("PaddleOCR-VL 1.6 loaded.")


def process_pdf(file_path: str):
    results = pipeline.predict(file_path)

    pages = []

    for page_number, result in enumerate(results, start=1):
        raw = result.json

        # PaddleOCR result may be a dict directly.
        res = raw.get("res", raw) if isinstance(raw, dict) else raw

        parsing_list = res.get("parsing_res_list", []) if isinstance(res, dict) else []

        blocks = []

        for block in parsing_list:
            content = block.get("block_content")

            if not isinstance(content, str):
                continue

            text = (
                content
                .replace("\r", "")
                .replace("\t", " ")
                .strip()
            )

            if not text:
                continue

            blocks.append({
                "order": block.get("block_order"),
                "label": block.get("block_label"),
                "text": text,
            })

        blocks.sort(
            key=lambda x: (
                x["order"] is None,
                x["order"] if x["order"] is not None else 0,
            )
        )

        page_text = "\n".join(block["text"] for block in blocks)

        pages.append({
            "page": page_number,
            "blocks": blocks,
            "text": page_text,
        })

    full_text = "\n\n".join(
        f"===== PAGE {page['page']} =====\n{page['text']}"
        for page in pages
        if page["text"]
    )

    return {
        "model": "PaddleOCR-VL-1.6",
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text,
    }
