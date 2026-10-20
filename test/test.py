import json
import sys
import numpy as np

from src.retrieval import Retriever


CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
CITATIONS_PATH = "data/citations.json"
OUTPUT_PATH = "test_output.txt"


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG), and why is it useful for large language models?",

    "What is HyDE (Hypothetical Document Embeddings), and how does it improve dense retrieval?",

    "How does Dense Passage Retrieval (DPR) perform passage retrieval for open-domain question answering?",

    "What are the main advantages of using retrieval-augmented generation compared with relying only on the language model's parametric knowledge?",

    "What is Self-RAG, and how does it use self-reflection during retrieval and generation?",

    "What problem does Corrective Retrieval-Augmented Generation (CRAG) attempt to solve, and how does it correct poor retrieval results?",

    "How does RAGAS evaluate a Retrieval-Augmented Generation system?",

    "What is the 'Lost in the Middle' problem, and how does it affect language models when using long contexts?",

    "How do RAG, REALM, and RETRO differ in the way they use external retrieved information?",

    # Intentionally outside the corpus
    "What is the current market capitalization of NVIDIA in September 2026?",
]


# ============================================================
# TEE OUTPUT
# ============================================================

class Tee:

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_citations():

    try:

        with open(
            CITATIONS_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except FileNotFoundError:

        print(
            "\nWARNING: data/citations.json was not found."
        )

        print(
            "All papers will receive a citation count of 0."
        )

        return {}


# ============================================================
# CREDIBILITY SCORE
# ============================================================

def calculate_credibility(
    citation_count,
    max_citations
):

    if max_citations <= 0:
        return 0.0

    if citation_count <= 0:
        return 0.0

    # Logarithmic normalization.
    #
    # This produces a score between 0 and 1 while
    # preventing extremely highly cited papers from
    # completely dominating the score.

    score = (
        np.log1p(citation_count)
        / np.log1p(max_citations)
    )

    return float(
        min(score, 1.0)
    )


def add_credibility_scores(
    results,
    citations
):

    # Extract valid citation counts from citation metadata.
    citation_values = []

    for value in citations.values():

        # New format:
        # {
        #     "citation_count": 726,
        #     "year": 2024,
        #     ...
        # }
        if isinstance(value, dict):
            citation_count = value.get(
                "citation_count",
                0
            )

        # Also support the old format:
        # "Paper Title": 726
        else:
            citation_count = value

        # Ignore missing / invalid citation counts.
        if citation_count is not None:
            try:
                citation_count = int(citation_count)

                if citation_count > 0:
                    citation_values.append(
                        citation_count
                    )

            except (TypeError, ValueError):
                pass

    # Find the maximum citation count for
    # logarithmic normalization.
    max_citations = (
        max(citation_values)
        if citation_values
        else 0
    )

    updated_results = []

    for result in results:

        paper_title = result["paper_title"]

        paper_metadata = citations.get(
            paper_title,
            {}
        )

        # Handle both dictionary and numeric formats.
        if isinstance(paper_metadata, dict):

            citation_count = paper_metadata.get(
                "citation_count",
                0
            )

        else:
            citation_count = paper_metadata

        # Handle null citation counts.
        if citation_count is None:
            citation_count = 0

        try:
            citation_count = int(citation_count)

        except (TypeError, ValueError):
            citation_count = 0

        credibility = calculate_credibility(
            citation_count,
            max_citations
        )

        result = result.copy()

        result["citation_count"] = (
            citation_count
        )

        result["credibility_score"] = (
            credibility
        )

        updated_results.append(result)

    return updated_results

# ============================================================
# BUILD PROMPT
# ============================================================

def build_test_prompt(
    question,
    results
):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

Paper:
{result['paper_title']}

Page:
{result['page']}

Semantic similarity:
{result['score']:.4f}

Citation count:
{result['citation_count']:,}

Citation-based credibility score:
{result['credibility_score']:.4f}

Evidence:
{result['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are a research-paper question answering assistant.

Answer the user's question using ONLY the supplied
research-paper evidence.

Each source has three important signals:

1. Semantic similarity
   - How relevant the retrieved text is to the question.

2. Citation count
   - The number of citations received by the paper.

3. Citation-based credibility score
   - A normalized score from 0 to 1 based on citation count.

IMPORTANT:

- Do NOT assume that a highly cited paper is automatically correct.
- Citation count represents scholarly impact, not factual correctness.
- Prefer evidence that is directly relevant to the question.
- Use credibility as an additional source-quality signal.
- Do not let citation count override clearly more relevant evidence.
- Do not invent information.
- Do not use outside knowledge.
- If the supplied evidence is insufficient, explicitly say so.
- Cite important claims using the paper title and page number.
- When useful, explain which source was considered more credible and why.
- Distinguish between evidence from different papers.

USER QUESTION:
{question}

RETRIEVED RESEARCH-PAPER EVIDENCE:
{context}

Now answer the question concisely but technically.

At the end, provide a short:

SOURCE ASSESSMENT

For the sources actually used in your answer, mention:
- paper
- page
- citation count
- credibility score
- why the source was useful
"""

    return prompt


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_test_answer(
    question,
    results
):

    from src.generation import (
        client,
        MODEL_NAME
    )

    prompt = build_test_prompt(
        question,
        results
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# PRINT RETRIEVAL RESULTS
# ============================================================

def print_retrieval_results(
    results
):

    print("\n")
    print("-" * 90)
    print("TOP RETRIEVED SOURCES")
    print("-" * 90)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\nSOURCE {rank}")

        print(
            f"Paper              : "
            f"{result['paper_title']}"
        )

        print(
            f"Page               : "
            f"{result['page']}"
        )

        print(
            f"Semantic similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Citation count     : "
            f"{result['citation_count']:,}"
        )

        print(
            f"Credibility score  : "
            f"{result['credibility_score']:.4f}"
        )

        preview = (
            result["text"]
            .replace("\n", " ")
        )

        if len(preview) > 300:
            preview = (
                preview[:300]
                + "..."
            )

        print(
            f"Preview            : "
            f"{preview}"
        )


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    retriever,
    citations,
    question,
    test_number
):

    print("\n")
    print("#" * 90)
    print(f"TEST {test_number}")
    print("#" * 90)

    print(
        f"\nQUESTION:\n{question}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = retriever.search(
        question,
        top_k=5
    )

    # --------------------------------------------------------
    # Credibility
    # --------------------------------------------------------

    results = add_credibility_scores(
        results,
        citations
    )

    # --------------------------------------------------------
    # Display retrieval results
    # --------------------------------------------------------

    print_retrieval_results(
        results
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    print("\n")
    print("-" * 90)
    print("GENERATED ANSWER")
    print("-" * 90)

    try:

        answer = generate_test_answer(
            question,
            results
        )

        print("\n")
        print(answer)

    except Exception as e:

        print(
            "\nGENERATION FAILED"
        )

        print(
            f"Error: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Open output file
    # --------------------------------------------------------

    output_file = open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    )

    # Duplicate stdout:
    #
    # Everything printed will appear both:
    #
    # 1. In the terminal
    # 2. In test_output.txt

    original_stdout = sys.stdout

    sys.stdout = Tee(
        original_stdout,
        output_file
    )

    try:

        print("\n")
        print("=" * 90)
        print("ADAPTIVE MULTIMODAL RAG - TEST SUITE")
        print("=" * 90)

        print(
            f"\nOutput is being saved to: "
            f"{OUTPUT_PATH}"
        )

        # ----------------------------------------------------
        # Load corpus
        # ----------------------------------------------------

        print("\nLoading corpus...")

        chunks = load_chunks()

        embeddings = np.load(
            EMBEDDINGS_PATH
        )

        citations = load_citations()

        print(
            f"Chunks loaded     : "
            f"{len(chunks)}"
        )

        print(
            f"Embeddings loaded : "
            f"{embeddings.shape}"
        )

        print(
            f"Citation records  : "
            f"{len(citations)}"
        )

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        print(
            "\nLoading retriever..."
        )

        retriever = Retriever(
            embeddings,
            chunks
        )

        print(
            "\nRetriever ready."
        )

        # ----------------------------------------------------
        # Run tests
        # ----------------------------------------------------

        print("\n")
        print("=" * 90)
        print(
            f"RUNNING "
            f"{len(TEST_QUESTIONS)} "
            f"TEST QUESTIONS"
        )
        print("=" * 90)

        for i, question in enumerate(
            TEST_QUESTIONS,
            start=1
        ):

            run_test(
                retriever,
                citations,
                question,
                i
            )

            if i < len(TEST_QUESTIONS):

                input(
                    "\nPress ENTER to run "
                    "the next test..."
                )

        print("\n")
        print("=" * 90)
        print("ALL TESTS COMPLETED")
        print("=" * 90)

        print(
            f"\nComplete output saved to: "
            f"{OUTPUT_PATH}"
        )

    finally:

        # Restore normal stdout
        sys.stdout = original_stdout

        # Close output file
        output_file.close()


if __name__ == "__main__":
    main()