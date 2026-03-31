from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    # We are now using the OCR output file
    inp = Path("output/multimodal/elements_ocr.jsonl")
    out = Path("output/multimodal/chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Missing {inp}. Run extract_multimodal.py and caption_images_local_ocr.py first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks: List[Dict[str, Any]] = []

    for line in inp.read_text(encoding="utf-8").splitlines():
        el: Dict[str, Any] = json.loads(line)

        content = (el.get("content") or "").strip()
        if not content:
            continue

        # Tables: keep as-is if small
        if el["element_type"] == "table" and len(content) < 4000:
            parts = [content]
        else:
            parts = splitter.split_text(content)

        for i, part in enumerate(parts):
            chunks.append({
                "id": f"{el['meta'].get('source','doc')}_p{el['page']}_{el['element_type']}_{el['index']}_c{i}",
                "text": f"[{el['element_type'].upper()}] page={el['page']}\n{part}",
                "meta": {
                    "element_type": el["element_type"],
                    "page": el["page"],
                    "element_index": el["index"],
                    **(el.get("meta") or {}),
                }
            })

    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
