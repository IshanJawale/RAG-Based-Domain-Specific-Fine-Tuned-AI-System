"""
ingestion.py

PDF text + figure extraction.

TWO figure extraction modes:
  1. Embedded raster images  (get_images)
     - Only raster bitmaps actually embedded in the PDF (photos, pre-rendered diagrams)
     - Filtered by: minimum size, aspect ratio, content (rejects icons/arrows/boxes)

  2. Page-rendered figures  (get_pixmap with clip)
     - For every page whose text mentions "Figure X", renders that page region
       into a PNG.  This captures VECTOR diagrams (flowcharts, architecture
       diagrams drawn with lines/shapes) that get_images() MISSES entirely.
     - These are saved alongside embedded images.

Content filtering (both modes):
  - size filter : skip images smaller than MIN_EMBED_PX in either dimension
  - aspect ratio: skip very thin strips (ratio > 8) — lines, rule, arrows
  - white check : skip if >93% of pixels are near-white (blank page slice)
  - dark check  : skip if >93% of pixels are near-black (solid dark box)
  - std check   : skip if pixel standard deviation < 12 (near solid colour)
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pymupdf

# Pillow is used only for embedded-raster analysis (lightweight, no GPU)
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    print("[ingestion] Pillow not found — image quality filtering disabled for embedded rasters.")


PAPERS_DIR  = Path("papers")
DATA_DIR    = Path("data")
FIGURES_DIR = DATA_DIR / "figures"

# ── Tuning knobs ─────────────────────────────────────────────────────────────
MIN_EMBED_PX   = 200   # min width AND height for embedded raster images (px)
MIN_RENDER_PX  = 180   # min width AND height for rendered page-crops (px)
MAX_ASPECT     = 8.0   # skip if max_dim / min_dim > this  (lines, arrows)
MAX_MONO_RATIO = 0.93  # skip if >93% pixels near-white or near-black
MIN_PIX_STD    = 12    # skip if pixel std < 12 (solid-colour box)
RENDER_ZOOM    = 2.0   # page render zoom factor (2× → ~144 Dpi at 72 Dpi base)
# ─────────────────────────────────────────────────────────────────────────────

# Caption pattern: "Figure 2.", "Fig. 3:", "Fig 4 ", etc.
_CAPTION_RE = re.compile(
    r"(?:Figure|Fig\.?)\s+(\d+)[.:)–\-]?\s*([^\n]{0,200})",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Text helpers
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+",       " ",   text)
    text = re.sub(r"\n\s*\n+",  "\n\n",   text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def _page_captions(page_text: str) -> dict[int, str]:
    """Return {figure_number: caption_text} for all captions on this page."""
    caps = {}
    for m in _CAPTION_RE.finditer(page_text):
        n = int(m.group(1))
        if n not in caps:
            caps[n] = m.group(2).strip()
    return caps


# ══════════════════════════════════════════════════════════════════════════════
# Image quality filter
# ══════════════════════════════════════════════════════════════════════════════

def _is_useful_image_bytes(image_bytes: bytes, width: int, height: int) -> bool:
    """
    Return True if the image looks like a real figure (not icon/arrow/box/blank).
    Uses PIL + numpy for fast pixel analysis.
    """
    # ── Dimension checks ──────────────────────────────────────────────────────
    if width < MIN_EMBED_PX or height < MIN_EMBED_PX:
        return False

    if min(width, height) == 0:
        return False

    aspect = max(width, height) / min(width, height)
    if aspect > MAX_ASPECT:
        return False   # extremely thin strip — likely a rule or arrow

    # ── Content checks ────────────────────────────────────────────────────────
    if not _PIL_AVAILABLE:
        return True    # can't analyse pixels; allow through

    try:
        img   = PILImage.open(io.BytesIO(image_bytes)).convert("L")
        arr   = np.asarray(img, dtype=np.float32)
        total = arr.size

        white_ratio = float(np.sum(arr > 240)) / total
        dark_ratio  = float(np.sum(arr <  15)) / total
        pix_std     = float(np.std(arr))

        if white_ratio > MAX_MONO_RATIO:   return False  # blank / white box
        if dark_ratio  > MAX_MONO_RATIO:   return False  # solid dark box
        if pix_std     < MIN_PIX_STD:      return False  # near solid colour

        return True

    except Exception:
        return False   # corrupted — skip


def _is_useful_pixmap(pix: pymupdf.Pixmap) -> bool:
    """
    Same quality check for a PyMuPDF Pixmap (used for rendered figures).
    Converts to PNG bytes and delegates to _is_useful_image_bytes.
    """
    if pix.width < MIN_RENDER_PX or pix.height < MIN_RENDER_PX:
        return False

    aspect = max(pix.width, pix.height) / max(min(pix.width, pix.height), 1)
    if aspect > MAX_ASPECT:
        return False

    # Quick numpy check without PIL
    try:
        # samples is a bytes buffer of raw pixel data
        arr   = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
        gray  = arr.mean(axis=2).astype(np.float32)
        total = gray.size

        white_ratio = float(np.sum(gray > 240)) / total
        dark_ratio  = float(np.sum(gray <  15)) / total
        pix_std     = float(np.std(gray))

        if white_ratio > MAX_MONO_RATIO: return False
        if dark_ratio  > MAX_MONO_RATIO: return False
        if pix_std     < MIN_PIX_STD:    return False

        return True
    except Exception:
        return pix.width >= MIN_RENDER_PX and pix.height >= MIN_RENDER_PX


# ══════════════════════════════════════════════════════════════════════════════
# Extraction Mode 1 — embedded raster images
# ══════════════════════════════════════════════════════════════════════════════

def _extract_embedded(
    doc:         pymupdf.Document,
    page:        pymupdf.Page,
    page_number: int,
    paper_id:    str,
    captions:    dict[int, str],
    counter:     dict,           # mutable counter shared across calls
) -> list[dict]:
    """Extract embedded bitmap images from a single page."""

    figures   = []
    img_list  = page.get_images(full=True)

    for image_info in img_list:
        xref = image_info[0]
        try:
            img_data  = doc.extract_image(xref)
            img_bytes = img_data["image"]
            img_ext   = img_data["ext"]
            width     = img_data.get("width",  0)
            height    = img_data.get("height", 0)

            if not _is_useful_image_bytes(img_bytes, width, height):
                continue

            counter[page_number] = counter.get(page_number, 0) + 1
            idx  = counter[page_number]
            name = f"{paper_id}_page_{page_number}_figure_{idx}.{img_ext}"
            path = FIGURES_DIR / name

            with open(path, "wb") as f:
                f.write(img_bytes)

            # Best-effort caption: figure index on page ≈ figure number
            caption = captions.get(idx, "")
            # Also check neighbouring page captions (caption may be on next page)
            if not caption:
                caption = captions.get(idx + 1, "")

            figures.append({
                "figure_id":    f"{paper_id}_page_{page_number}_figure_{idx}",
                "paper_id":     paper_id,
                "paper_title":  paper_id,
                "page":         page_number,
                "width":        width,
                "height":       height,
                "caption":      caption,
                "path":         str(path),
                "source":       "embedded",
            })

        except Exception as e:
            print(f"  [embed] skip xref={xref} page={page_number}: {e}")

    return figures


# ══════════════════════════════════════════════════════════════════════════════
# Extraction Mode 2 — page-rendered crops (captures vector figures)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rendered(
    page:        pymupdf.Page,
    page_number: int,
    paper_id:    str,
    page_text:   str,
    captions:    dict[int, str],   # {fig_num: caption} on this page
    all_captions: dict[int, dict[int, str]],  # page_number → {fig_num: caption}
) -> list[dict]:
    """
    For each figure caption found on this page, render the page region
    that likely contains the figure and save it as PNG.

    Returns list of figure metadata dicts.
    """

    if not captions:
        return []

    figures    = []
    page_rect  = page.rect
    mat        = pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)

    # Find bounding boxes of caption text on the page using text blocks
    # get_text("blocks") → list of (x0, y0, x1, y1, text, block_no, type)
    text_blocks = page.get_text("blocks")

    def _find_caption_rect(fig_num: int) -> pymupdf.Rect | None:
        """Return bounding rect of the 'Figure N' text block, if found."""
        for blk in text_blocks:
            if blk[6] != 0:         # skip non-text blocks
                continue
            blk_text = blk[4]
            for m in _CAPTION_RE.finditer(blk_text):
                if int(m.group(1)) == fig_num:
                    return pymupdf.Rect(blk[0], blk[1], blk[2], blk[3])
        return None

    # Sort figure numbers so we can use neighbours to bound clipping
    sorted_figs = sorted(captions.keys())

    for i, fig_num in enumerate(sorted_figs):

        caption_rect = _find_caption_rect(fig_num)

        if caption_rect is None:
            # Caption text found in get_text() but not locatable as a block.
            # Use a full half-page crop as fallback.
            clip_top    = page_rect.y0 + (page_rect.height * 0.1)
            clip_bottom = page_rect.y0 + (page_rect.height * 0.7)
        else:
            caption_top = caption_rect.y0

            # Upper bound: either page top or bottom of previous figure's caption
            if i > 0:
                prev_fig_num   = sorted_figs[i - 1]
                prev_cap_rect  = _find_caption_rect(prev_fig_num)
                clip_top       = prev_cap_rect.y1 + 5 if prev_cap_rect else page_rect.y0
            else:
                clip_top = page_rect.y0

            # Lower bound: bottom of the caption text + small margin
            clip_bottom = caption_rect.y1 + 8

            # Safety: figure region must be at least 100pt tall
            if (clip_bottom - clip_top) < 100:
                clip_top = max(page_rect.y0, caption_top - 300)

        clip = pymupdf.Rect(
            page_rect.x0,
            clip_top,
            page_rect.x1,
            clip_bottom,
        )

        try:
            pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=pymupdf.csRGB)
        except Exception as e:
            print(f"  [render] skip fig {fig_num} page {page_number}: {e}")
            continue

        if not _is_useful_pixmap(pix):
            continue

        name = f"{paper_id}_page_{page_number}_rendered_fig{fig_num}.png"
        path = FIGURES_DIR / name

        try:
            pix.save(str(path))
        except Exception as e:
            print(f"  [render] save failed fig {fig_num} page {page_number}: {e}")
            continue

        # Caption: from this page, or check neighbouring pages too
        caption = captions.get(fig_num, "")
        if not caption:
            # Check adjacent pages
            for dp in (1, -1, 2):
                adj_caps = all_captions.get(page_number + dp, {})
                if fig_num in adj_caps:
                    caption = adj_caps[fig_num]
                    break

        figures.append({
            "figure_id":    f"{paper_id}_page_{page_number}_rendered_fig{fig_num}",
            "paper_id":     paper_id,
            "paper_title":  paper_id,
            "page":         page_number,
            "width":        pix.width,
            "height":       pix.height,
            "caption":      caption,
            "path":         str(path),
            "source":       "rendered",
        })

    return figures


# ══════════════════════════════════════════════════════════════════════════════
# Per-PDF extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Extract text + figures from a single PDF.

    Returns (pages, figures).
    Each page dict   : {paper_id, paper_title, page, text}
    Each figure dict : {figure_id, paper_id, paper_title, page,
                        width, height, caption, path, source}
    """

    doc      = pymupdf.open(pdf_path)
    paper_id = pdf_path.stem
    pages    = []
    all_figs = []

    # ── Pass 1: extract text + build caption maps ──────────────────────────
    text_by_page: dict[int, str]             = {}
    caps_by_page: dict[int, dict[int, str]]  = {}

    for pg_idx, page in enumerate(doc):
        page_num = pg_idx + 1
        text     = clean_text(page.get_text("text"))
        text_by_page[page_num] = text
        caps_by_page[page_num] = _page_captions(text)
        if text:
            pages.append({
                "paper_id":    paper_id,
                "paper_title": pdf_path.stem,
                "page":        page_num,
                "text":        text,
            })

    # ── Pass 2: extract figures ────────────────────────────────────────────
    embed_counter: dict[int, int] = {}   # page_number → count of embedded imgs

    for pg_idx, page in enumerate(doc):
        page_num    = pg_idx + 1
        page_caps   = caps_by_page.get(page_num, {})

        # Mode 1: embedded rasters
        embedded = _extract_embedded(
            doc, page, page_num, paper_id, page_caps, embed_counter
        )
        all_figs.extend(embedded)

        # Mode 2: rendered crops (for pages with figure captions)
        if page_caps:
            rendered = _extract_rendered(
                page, page_num, paper_id,
                text_by_page.get(page_num, ""),
                page_caps,
                caps_by_page,
            )
            all_figs.extend(rendered)

    # Fix paper_title (was set to paper_id stem; use full name)
    for fig in all_figs:
        fig["paper_title"] = pdf_path.stem

    doc.close()
    return pages, all_figs


# ══════════════════════════════════════════════════════════════════════════════
# Corpus ingestion
# ══════════════════════════════════════════════════════════════════════════════

def ingest_corpus() -> tuple[list[dict], list[dict]]:
    """Process all PDFs in papers/. Saves figures.json."""

    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDFs found in papers/")

    all_pages:   list[dict] = []
    all_figures: list[dict] = []
    n_embedded  = 0
    n_rendered  = 0

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        pages, figures = extract_pdf(pdf_path)
        all_pages.extend(pages)
        all_figures.extend(figures)
        n_e = sum(1 for f in figures if f.get("source") == "embedded")
        n_r = sum(1 for f in figures if f.get("source") == "rendered")
        n_embedded += n_e
        n_rendered += n_r
        print(f"          pages={len(pages)}  embedded={n_e}  rendered={n_r}")

    with open(DATA_DIR / "figures.json", "w", encoding="utf-8") as f:
        json.dump(all_figures, f, indent=2, ensure_ascii=False)

    print("\n" + "-" * 50)
    print("Ingestion complete")
    print(f"  Pages extracted  : {len(all_pages)}")
    print(f"  Embedded figures : {n_embedded}  (bitmap images, quality-filtered)")
    print(f"  Rendered figures : {n_rendered}  (vector/composite, page-rendered)")
    print(f"  Total figures    : {len(all_figures)}")
    print("-" * 50 + "\n")

    return all_pages, all_figures