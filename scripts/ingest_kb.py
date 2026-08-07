"""Ingest a knowledge-base document for one business.

Reads a text file, splits it into chunks, embeds via MiniMax,
saves into VectorStore on disk.

Usage:
    uv run python scripts/ingest_kb.py \
        --business default \
        --file kb/default/faq.txt \
        --source faq
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from callmind.brain import VectorStore, chunk_text
from callmind.config import get_settings
from callmind.llm.embeddings import MinimaxEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ingest_kb")


async def run(business_id: str, path: Path, source: str, batch_size: int) -> None:
    s = get_settings()
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    log.info("split %s into %d chunks", path, len(chunks))

    store = VectorStore(business_id, s.kb_dir)
    if not store.is_empty():
        log.info("loaded existing index with %d chunks", len(store._chunks))

    emb = MinimaxEmbeddings(
        api_key=s.llm_api_key,
        base_url=s.llm_base_url,
        endpoint=s.embedding_endpoint,
        model=s.embedding_model,
        embedding_type=s.embedding_type,
    )
    try:
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = await emb.embed(batch)
            vectors.extend(vecs)
            log.info("embedded %d/%d", len(vectors), len(chunks))
    finally:
        await emb.close()

    store.add(chunks, vectors, source=source or path.name)
    store.save()
    log.info("saved index with %d chunks at %s", len(store._chunks), store.dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a KB doc into CallMind vector store")
    parser.add_argument("--business", required=True, help="business_id")
    parser.add_argument("--file", required=True, type=Path, help="path to .txt file")
    parser.add_argument("--source", default="", help="source tag stored with each chunk")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-append", action="store_true")
    args = parser.parse_args()
    if not args.file.exists():
        raise SystemExit(f"file not found: {args.file}")
    asyncio.run(run(args.business, args.file, args.source, args.batch_size))


if __name__ == "__main__":
    main()