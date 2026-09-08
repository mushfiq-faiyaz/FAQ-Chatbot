"""
RAG Query Pipeline — Steps 5-7: Retrieve -> Generate -> Cite Sources

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
ingest.py built the database. test_query.py proved we can find relevant
chunks in it. This script is where it finally becomes a real "chatbot":

    your question
        --> RETRIEVE the most relevant chunks from Chroma (same as test_query.py)
        --> GENERATE an answer by handing those chunks + your question to an LLM
        --> CITE which source document(s) the answer's information came from

This is the "RAG" part of Retrieval-Augmented Generation in full: we're
"augmenting" what the LLM can say by "retrieving" real facts from your
docs FIRST, instead of letting it just guess from what it was trained on.
That's what keeps a support chatbot honest — it can't make up a refund
policy that doesn't exist in your actual documents.

Usage:
    pip install groq python-dotenv --break-system-packages
    (also needs chromadb + sentence-transformers, already installed)

    Get a free API key at console.groq.com (no credit card required),
    then put it in a .env file in this same folder:
        GROQ_API_KEY=your_groq_key_here

    Then run:
        python rag_chat.py --db_dir ./chroma_db --query "How do I get a refund?"
"""

import argparse
import os

# load_dotenv() reads a ".env" file in your project folder and copies its
# key=value lines into the environment variables for this Python process.
# WHY DO THIS INSTEAD OF PASTING YOUR KEY DIRECTLY IN THE CODE?
# If your API key is typed directly into rag_chat.py and you ever share
# this file, push it to GitHub, or show it to a client as a demo, your
# key leaks with it. Keeping it in a separate .env file (which you never
# commit/share) means the actual script is safe to show anyone.
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

# The official Groq Python SDK. Groq is a cloud API (not local!) that
# runs open-source models (Llama, etc.) on custom hardware built for
# speed. It's genuinely free — no credit card required — for a generous
# daily request limit, which is why we're using it here instead of a
# paid API. The SDK's interface deliberately looks almost identical to
# Anthropic's/OpenAI's, which is why swapping providers only touches this
# import and the generate_answer() function below — nothing else in the
# pipeline (retrieval, chunk labeling, citations) needs to change.
from groq import Groq


# ==========================================================================
# STEP 5: RETRIEVE — pull the most relevant chunks out of the vector DB
# ==========================================================================
# This function is almost identical to what test_query.py already did.
# That's intentional — test_query.py WAS us proving this piece works in
# isolation before wiring it into the full pipeline.

def retrieve_chunks(query: str, db_dir: str, collection_name: str, top_k: int) -> list:
    """
    Search the Chroma database for the top_k chunks most similar in
    meaning to `query`. Returns a list of dicts: {text, source, chunk_index, distance}
    """
    # Same embedding model as ingest.py used — this has to match, or the
    # question's vector and the stored chunks' vectors won't be comparable
    # (see the detailed note on this in test_query.py's comments).
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(collection_name, embedding_function=embed_fn)

    results = collection.query(query_texts=[query], n_results=top_k)

    # We reshape Chroma's slightly awkward "list of lists" result format
    # into a clean list of plain dicts — much easier to work with in the
    # rest of this script (and easier to read/debug if you print it).
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return chunks


# ==========================================================================
# STEP 6: GENERATE — hand the retrieved chunks + question to Claude
# ==========================================================================

def build_context_block(chunks: list) -> str:
    """
    Format retrieved chunks into a single block of text to hand the LLM,
    with clear labels so it (and we) know which chunk came from where.

    WHY LABEL EACH CHUNK LIKE THIS?
    If we just dumped raw chunk text at the model with no labels, it would
    have no way to tell us which source a fact came from when we ask it to
    cite one — it would have to guess. Giving each chunk a numbered,
    labeled marker like "[Source 1: TaskFlow_Refund_Usage_Policy.pdf]"
    lets us explicitly instruct the model to reference these exact labels
    in its answer.
    """
    blocks = []
    for i, c in enumerate(chunks):
        # enumerate starts at 0, but "Source 0" reads oddly to a human —
        # i + 1 just shifts the displayed numbering to start at 1.
        label = f"[Source {i + 1}: {c['source']}]"
        blocks.append(f"{label}\n{c['text']}")
    # "\n\n---\n\n" is a visual divider between chunks — this isn't magic
    # syntax, just a human/model-readable separator so chunk boundaries
    # are obvious in the final combined string.
    return "\n\n---\n\n".join(blocks)


def build_system_prompt() -> str:
    """
    The system prompt sets the LLM's ROLE and RULES for this task. It is
    sent once per request, separately from the actual question, and
    steers behavior for the whole conversation turn.

    WHY THESE SPECIFIC RULES?
    - "ONLY use the provided context" stops the model from filling gaps
      with plausible-sounding guesses (a huge risk for a support bot —
      you don't want it inventing a refund policy that isn't real).
    - "Say so if the answer isn't in the context" gives it an honest
      escape hatch instead of hallucinating when the docs don't cover
      something.
    - "Cite using [Source N]" is what makes step 7 (citations) possible —
      we're asking the model to tag its own claims as it writes them.
    - Rule 4 originally just said "keep answers short and direct." That's
      a VAGUE instruction — the model has to guess what "short" means,
      and when the source material itself was bullet-pointed policy
      tiers, it decided a structured, multi-line answer was "concise
      enough." Vague instructions get inconsistent results. Giving it a
      concrete constraint (a sentence count, no headers/bullets) removes
      the guesswork and produces answers that actually fit a chat bubble.
    """
    return (
        "You are a helpful customer support assistant for TaskFlow. "
        "Answer the user's question using ONLY the information in the "
        "provided context below. Do not use any outside knowledge.\n\n"
        "Rules:\n"
        "1. If the answer is fully or partially covered in the context, "
        "answer clearly and concisely.\n"
        "2. If the context does not contain the answer, say so honestly "
        "instead of guessing.\n"
        "3. When you state a fact from the context, cite it inline using "
        "the matching label, e.g. [Source 1].\n"
        "4. This is a live chat bubble, not a document. Write your answer "
        "as plain conversational prose in 2-4 sentences MAXIMUM. Do NOT "
        "use markdown headers, bold text, or bullet-point lists. If the "
        "context has multiple conditions (e.g. different refund tiers), "
        "summarize the most likely relevant one in a sentence and briefly "
        "mention that other cases exist, rather than listing every case."
    )


def generate_answer(query: str, chunks: list, model: str = "openai/gpt-oss-120b") -> str:
    """
    Send the question + retrieved context to Groq's hosted model and
    return its written answer as a plain string.

    WHY "openai/gpt-oss-120b"?
    This is an open-weight 120-billion-parameter model that Groq hosts
    for free. Groq's model lineup changes fairly often as they add new
    hosted models and retire old ones (llama-3.3-70b-versatile, which
    this project used originally, was deprecated on Aug 16, 2026) — if
    this model ever stops working too, check console.groq.com/docs/models
    for Groq's current list and swap the string here. Nothing else in
    this file needs to change when you do.
    """
    # Groq() creates the client object. Like the Anthropic SDK, it
    # automatically looks for an API key in an environment variable —
    # for Groq that's GROQ_API_KEY, which load_dotenv() populated for us
    # from the .env file at the top of main(). We don't pass the key in
    # by hand here either.
    client = Groq()

    context_block = build_context_block(chunks)

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )

    # NOTE ON API SHAPE: Groq's SDK follows the same "chat completions"
    # pattern OpenAI popularized, which is why this looks slightly
    # different from the Anthropic call it replaced:
    #   - The system prompt is just another item in the `messages` list
    #     (role="system"), NOT a separate top-level `system=` argument
    #     like Anthropic uses.
    #   - The call is client.chat.completions.create(...) instead of
    #     client.messages.create(...).
    # Functionally, both are doing the exact same job: sending a system
    # instruction + a user message, and getting a written reply back.
    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_message},
        ],
    )

    # Groq/OpenAI-style responses nest the reply under
    # response.choices[0].message.content — "choices" because these APIs
    # technically support returning multiple alternative completions at
    # once (n > 1), though we're only requesting the default single one
    # here, so index [0] is always the one we want.
    return response.choices[0].message.content


# ==========================================================================
# STEP 7: CITE — tell the user which document(s) actually backed the answer
# ==========================================================================

def format_sources_used(chunks: list) -> str:
    """
    Build a human-readable "Sources" footer listing which documents were
    retrieved for this answer.

    NOTE: This lists every chunk that was RETRIEVED and shown to the
    model, not necessarily every chunk the model actually ended up citing
    in its written [Source N] tags. For a v1 demo this is good enough —
    it's honest about what was available, even if the model didn't quote
    every single one. A more advanced version could parse the model's
    [Source N] tags out of its answer text and only list the ones it
    actually used, but that's an optimization for later, not step 7's
    core requirement.
    """
    lines = []
    seen = set()
    for i, c in enumerate(chunks):
        # A single source PDF might supply more than one chunk to the
        # same answer — we don't want "TaskFlow_FAQ.pdf" listed twice,
        # so `seen` tracks which filenames we've already added.
        if c["source"] not in seen:
            lines.append(f"- [Source {i + 1}] {c['source']}")
            seen.add(c["source"])
    return "\n".join(lines)


# ==========================================================================
# MAIN — ties retrieve + generate + cite together for one question
# ==========================================================================

def answer_question(query: str, db_dir: str, collection_name: str = "taskflow_docs", top_k: int = 3) -> dict:
    """
    One-call convenience function: given a question, returns a dict with
    the answer and its sources. This is the function the Streamlit chat
    UI (step 8) will import and call directly — everything above it is
    internal plumbing.
    """
    chunks = retrieve_chunks(query, db_dir, collection_name, top_k)
    answer = generate_answer(query, chunks)
    sources = format_sources_used(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks}


def main():
    # load_dotenv() must run BEFORE we try to use the Anthropic client,
    # since that's what makes ANTHROPIC_API_KEY available to os.environ /
    # the anthropic SDK. Doing this at the very top of main() (before any
    # API calls happen) avoids a confusing "no API key found" error later.
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection_name", default="taskflow_docs")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # A friendly early check: if the API key genuinely isn't set (missing
    # .env file, typo'd variable name, etc.), fail with a clear message
    # here instead of letting the anthropic SDK throw a more cryptic
    # authentication error two steps later.
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found. Check your .env file.")
        return

    print(f"Question: {args.query}\n")

    print("=== STEP 5: Retrieving relevant chunks ===")
    result = answer_question(args.query, args.db_dir, args.collection_name, args.top_k)

    print("\n=== STEP 6: Generated Answer ===")
    print(result["answer"])

    print("\n=== STEP 7: Sources ===")
    print(result["sources"])


if __name__ == "__main__":
    main()