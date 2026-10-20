import json
import time
from pathlib import Path
from difflib import SequenceMatcher

import requests


print(">>> get_citations.py STARTED")


# ============================================================
# CONFIG
# ============================================================

PAPERS_DIR = Path("papers")
OUTPUT_PATH = Path("data/citations.json")

API_URL = "https://api.openalex.org/works"

REQUEST_DELAY = 1.0
MATCH_THRESHOLD = 0.85


# ============================================================
# TITLE NORMALIZATION
# ============================================================

def normalize_title(title):

    title = title.lower()

    for char in [
        ":", ",", ".", ";", "!", "?",
        "-", "_", "(", ")", "[", "]",
        "{", "}", "'", '"'
    ]:
        title = title.replace(char, " ")

    return " ".join(title.split())


def title_similarity(title_a, title_b):

    return SequenceMatcher(
        None,
        normalize_title(title_a),
        normalize_title(title_b)
    ).ratio()


# ============================================================
# FIND PAPERS
# ============================================================

def get_pdf_files():

    print(
        f">>> Looking for PDFs in: "
        f"{PAPERS_DIR.resolve()}"
    )

    files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    print(
        f">>> Found {len(files)} PDF files"
    )

    return files


# ============================================================
# LOAD EXISTING JSON
# ============================================================

def load_existing_results():

    if not OUTPUT_PATH.exists():

        print(
            ">>> No existing citations.json found"
        )

        return {}

    try:

        with open(
            OUTPUT_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        print(
            f">>> Loaded {len(data)} existing results"
        )

        return data

    except Exception as e:

        print(
            f">>> Could not load existing JSON: {e}"
        )

        return {}


# ============================================================
# SAVE JSON
# ============================================================

def save_results(results):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# EMPTY / FAILED RESULT
# ============================================================

def empty_result(
    match_status="not_found",
    year=None,
    matched_title=None,
    openalex_id=None,
    doi=None,
    similarity=0.0
):

    return {
        "citation_count": None,
        "year": year,
        "matched_title": matched_title,
        "openalex_id": openalex_id,
        "doi": doi,
        "title_similarity": similarity,
        "source": "OpenAlex",
        "match_status": match_status
    }


# ============================================================
# OPENALEX SEARCH
# ============================================================

def search_openalex(title):

    params = {
        "search": title,
        "per-page": 5,
        "select": (
            "id,"
            "display_name,"
            "publication_year,"
            "cited_by_count,"
            "doi,"
            "ids"
        )
    }

    print(
        "    Sending request to OpenAlex..."
    )

    response = requests.get(
        API_URL,
        params=params,
        timeout=20
    )

    print(
        f"    HTTP status: {response.status_code}"
    )

    response.raise_for_status()

    data = response.json()

    results = data.get(
        "results",
        []
    )

    print(
        f"    OpenAlex returned "
        f"{len(results)} results"
    )

    if not results:
        return None

    # --------------------------------------------------------
    # Find closest title
    # --------------------------------------------------------

    best = None
    best_score = 0.0

    for result in results:

        openalex_title = result.get(
            "display_name",
            ""
        )

        score = title_similarity(
            title,
            openalex_title
        )

        print(
            f"    Candidate: "
            f"{openalex_title}"
        )

        print(
            f"    Similarity: "
            f"{score:.3f}"
        )

        if score > best_score:

            best_score = score
            best = result

    if best is None:
        return None

    best["_title_similarity"] = best_score

    return best


# ============================================================
# PROCESS PAPER
# ============================================================

def process_paper(title):

    try:

        result = search_openalex(
            title
        )

    except Exception as e:

        print(
            f"    ❌ ERROR: {e}"
        )

        return empty_result(
            match_status="error"
        )

    # --------------------------------------------------------
    # No result
    # --------------------------------------------------------

    if result is None:

        print(
            "    ❌ No matching paper found"
        )

        return empty_result(
            match_status="not_found"
        )

    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    similarity = result.get(
        "_title_similarity",
        0.0
    )

    matched_title = result.get(
        "display_name"
    )

    citation_count = result.get(
        "cited_by_count"
    )

    year = result.get(
        "publication_year"
    )

    openalex_id = result.get(
        "id"
    )

    doi = result.get(
        "doi"
    )

    print(
        "\n    BEST MATCH:"
    )

    print(
        f"    {matched_title}"
    )

    print(
        f"    Citations: "
        f"{citation_count}"
    )

    print(
        f"    Year: "
        f"{year}"
    )

    print(
        f"    Title similarity: "
        f"{similarity:.3f}"
    )

    # --------------------------------------------------------
    # Reject low-confidence matches
    # --------------------------------------------------------

    if similarity < MATCH_THRESHOLD:

        print(
            f"    ⚠ REJECTED - "
            f"similarity below "
            f"{MATCH_THRESHOLD}"
        )

        return empty_result(
            match_status="low_confidence_match",
            year=year,
            matched_title=matched_title,
            openalex_id=openalex_id,
            doi=doi,
            similarity=similarity
        )

    # --------------------------------------------------------
    # Valid match
    # --------------------------------------------------------

    return {
        "citation_count": citation_count,
        "year": year,
        "matched_title": matched_title,
        "openalex_id": openalex_id,
        "doi": doi,
        "title_similarity": similarity,
        "source": "OpenAlex",
        "match_status": "matched"
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("OPENALEX CITATION LOOKUP")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not PAPERS_DIR.exists():

        print(
            f"❌ ERROR: papers directory does not exist:"
        )

        print(
            f"   {PAPERS_DIR.resolve()}"
        )

        return

    # --------------------------------------------------------
    # Get PDFs
    # --------------------------------------------------------

    pdf_files = get_pdf_files()

    if not pdf_files:

        print(
            "\n❌ No PDF files found."
        )

        return

    # --------------------------------------------------------
    # Test internet connection
    # --------------------------------------------------------

    print(
        "\n>>> Testing OpenAlex connection..."
    )

    try:

        response = requests.get(
            API_URL,
            params={
                "search": "Attention Is All You Need",
                "per-page": 1
            },
            timeout=10
        )

        print(
            f">>> Connection test HTTP status: "
            f"{response.status_code}"
        )

        if response.status_code != 200:

            print(
                "❌ OpenAlex connection test failed."
            )

            print(
                response.text[:500]
            )

            return

    except Exception as e:

        print(
            f"❌ Cannot connect to OpenAlex:"
        )

        print(
            f"   {e}"
        )

        return

    print(
        ">>> OpenAlex connection successful!"
    )

    # --------------------------------------------------------
    # Load previous results
    # --------------------------------------------------------

    old_citations = load_existing_results()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Build a NEW dictionary using ONLY the PDFs that
    # currently exist.
    #
    # This automatically removes duplicate/manual entries
    # from the old JSON.
    # --------------------------------------------------------

    citations = {}

    # --------------------------------------------------------
    # Process papers
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "PROCESSING PAPERS"
    )

    print(
        "=" * 80
    )

    for index, pdf_file in enumerate(
        pdf_files,
        start=1
    ):

        title = (
            pdf_file.stem
            .replace("_", " ")
            .strip()
        )

        print()
        print(
            f"[{index}/{len(pdf_files)}] "
            f"{title}"
        )

        # ----------------------------------------------------
        # Use previous result ONLY if it was successfully
        # matched.
        #
        # null / failed / low-confidence records are retried.
        # ----------------------------------------------------

        existing = old_citations.get(
            title
        )

        if (
            isinstance(existing, dict)
            and existing.get("match_status") == "matched"
            and existing.get("citation_count") is not None
        ):

            print(
                "    ✓ Already successfully processed"
            )

            citations[title] = existing

            continue

        # ----------------------------------------------------
        # Search OpenAlex
        # ----------------------------------------------------

        result = process_paper(
            title
        )

        citations[title] = result

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        save_results(
            citations
        )

        print(
            "\n    💾 Saved!"
        )

        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        if index < len(pdf_files):

            print(
                f"    Waiting "
                f"{REQUEST_DELAY} second..."
            )

            time.sleep(
                REQUEST_DELAY
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)

    matched = sum(
        1
        for x in citations.values()
        if (
            x.get("match_status") == "matched"
            and x.get("citation_count") is not None
        )
    )

    unresolved = len(citations) - matched

    print(
        f"\nTotal PDFs : {len(pdf_files)}"
    )

    print(
        f"Matched    : {matched}"
    )

    print(
        f"Unresolved : {unresolved}"
    )

    print(
        f"JSON file  : {OUTPUT_PATH.resolve()}"
    )

    print()
    print(
        ">>> get_citations.py FINISHED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        ">>> Calling main()..."
    )

    main()