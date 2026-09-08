"""
Quick sanity check for the vector store built by ingest.py.
This only tests retrieval (steps 1-4 output) — no LLM call yet.

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
ingest.py already built and saved the vector database. This script opens
that SAME database (it doesn't rebuild anything) and asks it: "given this
question, which stored chunks are the closest match?" That's exactly the
first half of what a real RAG chatbot does at answer-time — the only piece
missing here is actually sending those chunks to an LLM to generate a
written answer, which comes in step 6.

Usage:
    python test_query.py --db_dir ./chroma_db --query "How do I get a refund?"
"""

import argparse
import chromadb
from chromadb.utils import embedding_functions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection_name", default="taskflow_docs")
    # required=True means the script will refuse to run and print a
    # helpful error if you forget to pass --query — there's no sensible
    # default for "what question do you want to ask."
    parser.add_argument("--query", required=True)
    # How many of the closest-matching chunks to return. 3 is a common
    # starting point: enough context for an LLM to answer well later,
    # without flooding it with irrelevant chunks.
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # IMPORTANT: this MUST be the same model name used in ingest.py
    # (all-MiniLM-L6-v2). Every chunk in the database was turned into a
    # vector using that model's "understanding" of language. If you
    # embedded your QUESTION with a different model here, the resulting
    # vector would live in a totally different, incompatible "space" —
    # like comparing distances on two different maps that don't share a
    # scale. Same model in, same model out is required for the similarity
    # math to mean anything.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Connect to the exact same on-disk database folder that ingest.py
    # wrote to. This is a completely separate Python process/run from
    # ingest.py — the only thing connecting them is this shared db_dir
    # path, which is the whole point of a "persistent" database.
    client = chromadb.PersistentClient(path=args.db_dir)

    # get_collection (not create_collection!) opens the existing
    # collection ingest.py already built. If you typo the collection
    # name, or never ran ingest.py successfully, this line throws an
    # error telling you it doesn't exist — a useful early signal that
    # something upstream didn't finish correctly.
    collection = client.get_collection(args.collection_name, embedding_function=embed_fn)

    # This is the actual search step:
    # 1. Chroma runs your query text through embed_fn to get its vector.
    # 2. It compares that vector against every stored chunk's vector,
    #    using a distance measure (smaller distance = more similar).
    # 3. It returns the `top_k` closest matches.
    # query_texts takes a LIST (you could search multiple questions in
    # one call), which is why args.query is wrapped in [ ] even though
    # we're only asking one question here.
    results = collection.query(query_texts=[args.query], n_results=args.top_k)

    print(f"Query: {args.query}\n")

    # results is a dict where each value is a list-of-lists — the outer
    # list has one entry per query we asked (we only asked one, at index
    # [0]), and the inner list has one entry per result within that query.
    # zip(...) lets us walk through documents, metadatas, and distances
    # together in lockstep, so result #1's text lines up with result #1's
    # metadata and result #1's distance score, etc.
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        # meta['source'] and meta['chunk_index'] are exactly the metadata
        # fields we saved back in ingest.py's embed_and_store() — this is
        # them being put to use, telling us which PDF (and which chunk
        # number in it) this result came from.
        #
        # `dist` (distance) is a number showing how far apart the query's
        # vector and this chunk's vector are — LOWER means MORE similar.
        # It is NOT a percentage or a 0-1 confidence score; it depends on
        # the distance metric Chroma is using under the hood, so treat it
        # as "smaller is better," not as an absolute quality threshold.
        print(f"--- Result {i+1} (source: {meta['source']}, chunk {meta['chunk_index']}, distance: {dist:.4f}) ---")

        # Print only the first 300 characters of each result, so the
        # terminal output stays scannable — we're sanity-checking that
        # retrieval finds the RIGHT document, not reading the full answer
        # yet (that's what step 6's LLM call will do with the FULL text).
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
        print()


if __name__ == "__main__":
    main()