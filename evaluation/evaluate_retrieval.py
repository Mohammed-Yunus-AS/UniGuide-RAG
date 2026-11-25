from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uniguide.config import KNOWLEDGE_BASE_PATH, settings  # noqa: E402
from uniguide.retrieval import Retriever  # noqa: E402
from uniguide.generation import make_client  # noqa: E402
from uniguide.storage import load_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate UniGuide-RAG retrieval using labeled questions.")
    parser.add_argument("--csv", default=str(ROOT / "evaluation" / "benchmark_template.csv"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(KNOWLEDGE_BASE_PATH)
    if not chunks:
        print("Knowledge base is empty. Run the app and load the built-in demo documents first.")
        return 2

    rows = list(csv.DictReader(Path(args.csv).open(newline="", encoding="utf-8-sig")))
    rows = [r for r in rows if r.get("question", "").strip() and r.get("expected_source", "").strip()
            and not r["expected_source"].startswith("replace_with_")
            and r["expected_source"].strip().upper() not in {"NO_ANSWER", "UNANSWERABLE"}]
    if not rows:
        print("No answerable retrieval labels found. Add expected source filenames to the CSV.")
        return 2

    retriever = Retriever(chunks, client=make_client())
    reciprocal_ranks: list[float] = []
    hits = 0
    for row in rows:
        results = retriever.search(row["question"], args.k)
        expected = row["expected_source"].strip().casefold()
        rank = next((i for i, item in enumerate(results, start=1)
                     if expected in item.get("source", "").casefold()), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        print(f"Q: {row['question']}\n  Expected source: {row['expected_source']}\n  Rank: {rank or 'not in top-k'}")

    total = len(rows)
    print("\nRetrieval evaluation (source-level relevance labels)")
    print(f"Queries: {total}")
    print(f"HitRate@{args.k}: {hits / total:.4f}")
    print(f"MRR@{args.k}: {sum(reciprocal_ranks) / total:.4f}")
    print("Note: This evaluates whether the expected source appears in the top-k results; it does not measure answer correctness.")
    print("Rows marked NO_ANSWER / UNANSWERABLE are excluded from source-ranking metrics and should be evaluated separately for abstention quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
