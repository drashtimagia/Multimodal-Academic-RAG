"""
ingestion/extract_multimodal.py
--------------------------------
Extracts text, tables, and images from ALL PDFs in data/raw_pdfs/.
Each PDF gets its own image subdirectory to avoid filename collisions.
Output: output/multimodal/elements.jsonl  (all elements from all papers)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


@dataclass
class Element:
    element_type: str   # "text" | "table" | "image"
    page: int
    index: int
    content: str        # text OR markdown table OR placeholder for OCR
    meta: Dict[str, Any]


def _pixmap_to_png_bytes(pix: fitz.Pixmap) -> bytes:
    if pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png")


def extract_multimodal(pdf_path: Path, out_dir: Path) -> List[Element]:
    """Extract all elements from one PDF. Images go into out_dir/images/<pdf_stem>/."""
    img_dir = out_dir / "images" / pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    elements: List[Element] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # ── TEXT ──────────────────────────────────────────────────────────
        text = page.get_text("text").strip()
        if text:
            elements.append(Element(
                element_type="text",
                page=page_num,
                index=0,
                content=text,
                meta={"source": pdf_path.stem, "pdf_file": pdf_path.name},
            ))

        # ── TABLES ────────────────────────────────────────────────────────
        try:
            tables = page.find_tables()
            for t_i, table in enumerate(tables.tables):
                df = table.to_pandas()
                md = df.to_markdown(index=False)
                elements.append(Element(
                    element_type="table",
                    page=page_num,
                    index=t_i,
                    content=md,
                    meta={
                        "source": pdf_path.stem,
                        "pdf_file": pdf_path.name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                ))
        except Exception:
            pass

        # ── IMAGES ────────────────────────────────────────────────────────
        for i_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            png_bytes = _pixmap_to_png_bytes(pix)
            img_path = img_dir / f"p{page_num:03d}_img{i_i:02d}.png"
            img_path.write_bytes(png_bytes)
            elements.append(Element(
                element_type="image",
                page=page_num,
                index=i_i,
                content="",   # filled by caption_images_local_ocr.py
                meta={
                    "source": pdf_path.stem,
                    "pdf_file": pdf_path.name,
                    "image_path": str(img_path),
                },
            ))

    return elements


def save_jsonl(elements: List[Element], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for el in elements:
            f.write(json.dumps(asdict(el), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    PDF_DIR = Path("data/raw_pdfs")
    OUT_DIR = Path("output/multimodal")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/raw_pdfs/. Drop PDFs there first.")

    all_elements: List[Element] = []
    for p in pdfs:
        img_dir = OUT_DIR / "images" / p.stem
        if img_dir.exists() and any(img_dir.iterdir()):
            print(f"  >> Skipping: {p.name} (already extracted)")
            continue

        print(f"  Extracting: {p.name}")
        els = extract_multimodal(p, OUT_DIR)
        all_elements.extend(els)
        counts = {t: sum(1 for e in els if e.element_type == t) for t in ("text", "table", "image")}
        print(f"    -> text:{counts['text']}  tables:{counts['table']}  images:{counts['image']}")

    out = OUT_DIR / "elements.jsonl"
    save_jsonl(all_elements, out)
    print(f"\nOK  {len(all_elements)} elements from {len(pdfs)} PDF(s) -> {out}")
