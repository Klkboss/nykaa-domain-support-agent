"""Phase 9 verification (owner brief §7): the deployed persistent chroma_db/ index, READ-ONLY.

1. Its stored ids, texts and metadata equal the chunks of data/knowledge_base/policies.json (both strategies).
2. Repeated retrieval through search() (the answering pipeline) for every frozen query: the same results every round,
   and every returned chunk's id and text equal the KB chunk. Any exception (e.g. Chroma's InternalError
   "Error finding id", seen in the in-memory test index in Phase 8) is counted, not hidden.
The project writes nothing: no upsert, no delete, no rebuild (Chroma itself rewrites its index files on open, so
the stored vectors are compared with freshly computed ones as well). If the index does not match, this script reports it and stops.

Usage (socket guard):
  .venv\\Scripts\\python.exe -c "import tests.conftest as g, runpy; runpy.run_path('scripts/verify_persistent_chroma.py', run_name='__main__'); print('blocked network attempts:', g.blocked_attempts)"
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import app.core.config  # noqa: E402,F401
from app.core.config import ROOT  # noqa: E402
from app.rag.chunking import STRATEGIES, chunk_documents  # noqa: E402
from app.rag.documents import load_knowledge_base  # noqa: E402
from app.rag.embeddings import embed  # noqa: E402
from app.rag.retrieval import search  # noqa: E402
from app.rag.vector_store import CHROMA_DIR, COLLECTION_NAMES, persistent_client  # noqa: E402

ROUNDS = 50


def frozen_queries() -> list[str]:
    load = lambda name: json.loads((ROOT / "evaluation" / name).read_text(encoding="utf-8"))  # noqa: E731
    rag = load("rag_queries.json")
    rows = (rag["evaluation"] + rag["calibration"]["in_scope"] + rag["calibration"]["out_of_scope"]
            + [rag["out_of_scope_demo"]] + load("rag_intent_queries.json")["queries"]
            + load("rag_escalation_queries.json")["queries"] + load("test_queries.json")["queries"])
    return [r["query"] for r in rows]


def main() -> None:
    print(f"index: {CHROMA_DIR.relative_to(ROOT)} (read-only)")
    client = persistent_client()
    names = {c.name for c in client.list_collections()}
    queries = frozen_queries()
    ok = True
    for strategy in STRATEGIES:
        if COLLECTION_NAMES[strategy] not in names:  # never create it: get_collection() would
            print(f"[{strategy}] collection {COLLECTION_NAMES[strategy]} missing: NOT verified")
            ok = False
            continue
        collection = client.get_collection(COLLECTION_NAMES[strategy], embedding_function=None)
        expected = {c.chunk_id: c for c in chunk_documents(load_knowledge_base(), strategy)}
        stored = collection.get(include=["documents", "metadatas", "embeddings"])
        stored_map = {i: (d, m) for i, d, m in zip(stored["ids"], stored["documents"], stored["metadatas"])}
        same = (set(stored_map) == set(expected)
                and all(stored_map[i] == (c.text, c.metadata) for i, c in expected.items()))
        print(f"\n[{strategy}] {COLLECTION_NAMES[strategy]}: stored {len(stored_map)} chunks, KB gives {len(expected)}; "
              f"ids+texts+metadata identical: {same}")
        if same:  # the vectors too: Chroma rewrites its index files even on read-only use
            fresh = np.array(embed([expected[i].text for i in stored["ids"]]))
            drift = float(np.abs(np.array(stored["embeddings"]) - fresh).max())
            same = drift < 1e-5
            print(f"  stored embeddings vs freshly computed: max abs difference {drift:.2e} (ok: {same})")
        if not same:
            ok = False
            continue
        baseline, errors, mismatches, empty = None, Counter(), 0, 0
        for _ in range(ROUNDS):
            round_ids = []
            for q in queries:
                try:
                    hits = search(collection, q)
                except Exception as e:  # counted, not hidden
                    errors[f"{type(e).__name__}: {str(e)[:80]}"] += 1
                    round_ids.append(None)
                    continue
                empty += not hits
                mismatches += sum(h.chunk_id not in expected or h.text != expected[h.chunk_id].text for h in hits)
                round_ids.append([h.chunk_id for h in hits])
            baseline = baseline or round_ids
            if round_ids != baseline:
                mismatches += 1
        total = ROUNDS * len(queries)
        print(f"  {ROUNDS} rounds x {len(queries)} frozen queries = {total} searches")
        print(f"  exceptions: {sum(errors.values())} {dict(errors) or ''}")
        print(f"  empty results: {empty}   wrong id/text or round-to-round change: {mismatches}")
        ok &= not errors and not empty and not mismatches
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
