"""
chunking.py

Sentence-aware text chunker.

Instead of splitting on raw word boundaries (which cuts sentences in half),
this module first splits text into sentences using NLTK, then groups whole
sentences into chunks until the word budget is reached. Overlap is also done
at the sentence level, so every chunk starts on a sentence boundary.
"""

import json
import re
from pathlib import Path

import nltk

# Download the punkt tokenizer data on first run.
# Silent if already downloaded.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


DATA_DIR = Path("data")


# ------------------------------------------------------------------
# Low-level: split a block of text into sentences
# ------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """
    Split a block of text into individual sentences using NLTK.

    Falls back to a simple period-split if NLTK is unavailable.
    Empty strings are dropped.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: naive split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)

    # Drop blank sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


# ------------------------------------------------------------------
# Low-level: group sentences into overlapping chunks
# ------------------------------------------------------------------

def sentences_to_chunks(
    sentences: list[str],
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[str]:
    """
    Group sentences into overlapping chunks of approximately
    `chunk_size` words each.

    Overlap is handled by rewinding by approximately `overlap` words
    worth of sentences at the end of each chunk.

    Parameters
    ----------
    sentences  : list of sentence strings
    chunk_size : target maximum number of words per chunk
    overlap    : approximate number of words to overlap between chunks

    Returns
    -------
    list of chunk strings (each chunk is one or more complete sentences)
    """

    if not sentences:
        return []

    chunks = []
    start = 0

    while start < len(sentences):

        current_words = 0
        end = start

        # Greedily add sentences until we hit the word budget
        while end < len(sentences):

            sentence_words = len(sentences[end].split())

            # Always include at least one sentence per chunk
            if current_words > 0 and current_words + sentence_words > chunk_size:
                break

            current_words += sentence_words
            end += 1

        # Build the chunk text from sentences[start:end]
        chunk_text = " ".join(sentences[start:end]).strip()

        if chunk_text:
            chunks.append(chunk_text)

        # If we consumed all sentences, stop
        if end >= len(sentences):
            break

        # Rewind by ~overlap words (in whole sentences)
        rewind_words = 0
        rewind_index = end - 1

        while rewind_index > start and rewind_words < overlap:
            rewind_words += len(sentences[rewind_index].split())
            rewind_index -= 1

        # Move start forward, ensuring progress to avoid infinite loop
        new_start = max(rewind_index + 1, start + 1)
        start = new_start

    return chunks


# ------------------------------------------------------------------
# Main: chunk a list of extracted pages
# ------------------------------------------------------------------

def create_chunks(
    pages: list[dict],
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[dict]:
    """
    Convert a list of extracted pages into searchable, metadata-rich chunks.

    Each page dict must have keys:
        paper_id, paper_title, page, text

    Each output chunk dict has keys:
        chunk_id, paper_id, paper_title, page, text
    """

    all_chunks = []
    chunk_id = 0

    for page in pages:

        page_text = page.get("text", "").strip()

        if not page_text:
            continue

        sentences = split_into_sentences(page_text)

        page_chunks = sentences_to_chunks(
            sentences,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_text in page_chunks:

            all_chunks.append({
                "chunk_id":    chunk_id,
                "paper_id":    page["paper_id"],
                "paper_title": page["paper_title"],
                "page":        page["page"],
                "text":        chunk_text,
            })

            chunk_id += 1

    # Persist to disk
    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("--------------------------------")
    print("Chunking complete")
    print(f"Total chunks : {len(all_chunks)}")
    print(f"Saved to     : {output_path}")
    print("--------------------------------\n")

    return all_chunks