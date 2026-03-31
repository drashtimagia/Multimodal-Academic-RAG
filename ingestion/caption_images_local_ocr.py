from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import pytesseract


def ocr_image(image_path: Path) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """
    img = Image.open(image_path)

    # OCR config:
    # --psm 6 assumes a block of text (good default for documents/figures)
    # You can try --psm 4 or 11 if results are poor.
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")

    # Basic cleanup
    text = text.replace("\x0c", "").strip()
    return text


def main():
    inp = Path("output/multimodal/elements.jsonl")
    out = Path("output/multimodal/elements_ocr.jsonl")

    if not inp.exists():
        raise SystemExit(f"Missing: {inp}. Run extract_multimodal.py first.")

    lines = inp.read_text(encoding="utf-8").splitlines()
    out_lines: List[str] = []

    image_count = 0
    ocr_nonempty = 0

    for line in lines:
        el: Dict[str, Any] = json.loads(line)

        if el["element_type"] == "image":
            image_count += 1
            img_path = Path(el["meta"]["image_path"])

            if not img_path.exists():
                el["content"] = ""
                el["meta"]["ocr_error"] = f"image file not found: {img_path}"
            else:
                try:
                    text = ocr_image(img_path)
                    el["content"] = text
                    if text.strip():
                        ocr_nonempty += 1
                except Exception as e:
                    el["content"] = ""
                    el["meta"]["ocr_error"] = str(e)

        out_lines.append(json.dumps(el, ensure_ascii=False))

    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote -> {out}")
    print(f"Images found: {image_count}")
    print(f"Images with non-empty OCR text: {ocr_nonempty}")


if __name__ == "__main__":
    main()
