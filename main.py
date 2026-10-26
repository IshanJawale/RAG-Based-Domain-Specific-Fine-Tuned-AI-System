"""
main.py — CLI Entry Point for Indian Legal RAG Assistant

Usage:
    # 1. Ask a single question directly:
    python main.py "What is the procedure for anticipatory bail under Section 438 CrPC?"

    # 2. Use a specific model:
    python main.py --model Qwen3-VL-8B-Instruct "Explain Article 21 of the Indian Constitution"

    # 3. Interactive conversational REPL:
    python main.py
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.retrieval  import HybridRetriever
from src.reranking  import Reranker
from src.web_search import TavilySearcher
from src.pipeline    import run_query
from src.llm        import DEFAULT_MODEL, get_available_models

def main():
    parser = argparse.ArgumentParser(description="Indian Legal AI — RAG Assistant CLI")
    parser.add_argument("query", nargs="?", default=None, help="Legal question to ask")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=get_available_models(),
        help="Model to use for answer generation",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 70)
    print("[Indian Legal AI] Grounded Legal RAG & Assistant")
    print(f"[Model] Active: {args.model}")
    print("=" * 70 + "\n")

    print("Initializing federated legal retriever across LanceDB...")
    hybrid = HybridRetriever()

    print("Loading cross-encoder reranker...")
    reranker = Reranker()

    try:
        web_search = TavilySearcher()
    except Exception:
        web_search = None

    if args.query:
        result = run_query(
            question=args.query,
            hybrid=hybrid,
            reranker=reranker,
            web_search=web_search,
            verbose=True,
            model_name=args.model,
        )
        print("\n" + "=" * 70)
        print("FINAL ANSWER:\n")
        print(result["answer"])
        print("=" * 70 + "\n")
        return

    # Interactive mode
    print("Entered Interactive REPL Mode. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            query = input("\n⚖️ Ask a legal question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            result = run_query(
                question=query,
                hybrid=hybrid,
                reranker=reranker,
                web_search=web_search,
                verbose=True,
                model_name=args.model,
            )
            print("\n" + "=" * 70)
            print("FINAL ANSWER:\n")
            print(result["answer"])
            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
