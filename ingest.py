"""
RAG Ingestion Pipeline — Steps 1-4: Load -> Chunk -> Embed -> Store

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
A "RAG" (Retrieval-Augmented Generation) system needs a searchable database
of your documents BEFORE it can answer questions about them. This script
builds that database. It does NOT talk to an LLM at all — that comes later
in steps 5-8. Right now we're just prepping the data:

    PDF files  -->  raw text  -->  small text chunks  -->  number vectors  -->  saved to disk

Usage:
    pip install pdfplumber sentence-transformers chromadb --break-system-packages
    python ingest.py --docs_dir ./docs --db_dir ./chroma_db

Folder structure expected:
    docs/
        TaskFlow_Getting_Started_Guide.pdf
        TaskFlow_Pricing_Plans.pdf
        TaskFlow_Refund_Usage_Policy.pdf
        TaskFlow_Troubleshooting_Guide.pdf
        TaskFlow_FAQ.pdf
"""

# ---------- IMPORTS ----------
# argparse: lets you pass options on the command line, e.g. --docs_dir ./docs
#           instead of hardcoding paths in the script.
import argparse
# os: used here just to list files in a folder and build file paths safely
#     across Windows/Mac/Linux (os.path.join handles the \ vs / difference).
import os
# re: Python's regular-expression module. We use it to find patterns in text
#     (like "two newlines in a row" or "a sentence-ending period") so we can
#     split documents up intelligently instead of just cutting at a fixed
#     character count.
import re
# uuid: generates a random unique ID (like a fingerprint) for every chunk,
#       so each one has a distinct name when stored in the database.
import uuid

# pdfplumber: the library that actually opens a PDF and pulls the text out
#             of it. PDFs don't store "paragraphs" the way a Word doc does —
#             they store where each character sits on the page — so
#             pdfplumber has to reconstruct readable text from that.
import pdfplumber
# chromadb: the vector database itself. "Vector database" just means a
#           database built to store lists of numbers (vectors) and quickly
#           find which ones are most similar to each other.
import chromadb
# embedding_functions: a helper chromadb gives us so it knows HOW to turn
#           text into vectors automatically (using the model we choose)
#           whenever we add or query data.
from chromadb.utils import embedding_functions


# ==========================================================================
# STEP 1: LOAD — get plain text out of each PDF
# ==========================================================================

def clean_pdf_text(text: str) -> str:
    """
    Strip font-encoding artifacts pdfplumber sometimes leaves behind.

    WHY THIS EXISTS:
    Some PDFs use custom fonts for bullet points or special characters.
    When pdfplumber can't map a character to a normal letter, it leaves
    a placeholder like "(cid:127)" in the text instead (cid = "character ID",
    an internal font-table reference). Left alone, this junk would end up
    inside your chunks and eventually get shown to a user as garbled text.
    """
    # re.sub(pattern, replacement, text) means:
    # "find every match of `pattern` inside `text`, and replace it with
    #  `replacement`, then return the new string."
    #
    # Pattern breakdown: r"\(cid:\d+\)"
    #   \(         -> a literal "(" character (backslash "escapes" it,
    #                 because a bare "(" has special meaning in regex)
    #   cid:       -> the literal text "cid:"
    #   \d+        -> one or more digits (0-9), e.g. "127"
    #   \)         -> a literal ")" character
    # So this matches things like "(cid:127)", "(cid:5)", "(cid:9999)", etc.
    # We replace each match with a plain hyphen "-", which is a reasonable
    # stand-in for what was probably a bullet point.
    text = re.sub(r"\(cid:\d+\)", "-", text)
    return text


def load_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a single PDF, page by page."""
    # We build the full document text piece by piece (one string per page)
    # and join them at the end — this is faster than repeatedly doing
    # `full_text += page_text` in a loop, because string concatenation in
    # a loop rebuilds the whole string every time (slow for long documents).
    text_parts = []

    # pdfplumber.open(...) opens the PDF file. Using "with" here means
    # Python automatically closes the file when we're done with it (even
    # if an error happens partway through) — you don't have to remember
    # to call pdf.close() yourself.
    with pdfplumber.open(pdf_path) as pdf:
        # pdf.pages is a list-like object — one entry per page in the PDF.
        for page in pdf.pages:
            # page.extract_text() reads the page and returns its text as
            # one string, with pdfplumber's best guess at where lines
            # break. If a page has NO extractable text (e.g. it's a scanned
            # image with no text layer), this returns None instead of "" —
            # so `or ""` swaps None for an empty string to keep things safe.
            page_text = page.extract_text() or ""
            # Clean up any (cid:xxx) junk before we store this page's text.
            text_parts.append(clean_pdf_text(page_text))

    # Join every page's text together, separated by a newline, so the
    # final result reads like one continuous document.
    return "\n".join(text_parts)


def load_all_pdfs(docs_dir: str) -> dict:
    """Returns {filename: full_text} for every PDF in docs_dir."""
    # A dict (dictionary) here maps filename -> that file's extracted text.
    # This keeps track of WHICH document each piece of text came from,
    # which matters later for citing sources back to the user.
    docs = {}

    # os.listdir(docs_dir) returns every file/folder name inside docs_dir,
    # as plain strings (not full paths).
    for fname in os.listdir(docs_dir):
        # .lower() makes the check case-insensitive, so "Report.PDF" still
        # counts. .endswith(".pdf") filters out anything that isn't a PDF
        # (e.g. if a stray .txt or .DS_Store file were sitting in that
        # folder, we skip it instead of crashing on it).
        if fname.lower().endswith(".pdf"):
            # os.path.join safely builds "docs/TaskFlow_FAQ.pdf" (or the
            # Windows equivalent with backslashes) without you having to
            # worry about which OS you're running on.
            path = os.path.join(docs_dir, fname)
            docs[fname] = load_pdf_text(path)
            # A simple progress print so you can see it's working as it
            # processes each file, and roughly how much text came out.
            print(f"Loaded {fname} ({len(docs[fname])} chars)")
    return docs


# ==========================================================================
# STEP 2: CHUNK — break each document into small, focused pieces
# ==========================================================================
#
# WHY CHUNK AT ALL?
# When you later ask a question, the system will search for the chunk(s)
# most relevant to your question and hand ONLY those to the LLM — not the
# whole document. Two reasons this matters:
#   1. LLMs have a limited context window; you can't always stuff every
#      document in.
#   2. Smaller, focused chunks embed into more precise vectors, so search
#      results are more accurate. A giant "chunk" that covers 5 different
#      topics produces a vague, blended vector that doesn't match anything
#      well.
#
# WHY 400 WORDS WITH AN 80-WORD OVERLAP?
# 400 words is a rule-of-thumb size: big enough to hold a full explanation,
# small enough to stay focused on one topic. The 80-word overlap means the
# END of one chunk is repeated at the START of the next chunk. This exists
# so that if an important sentence happens to sit right at a chunk boundary,
# it still appears in full inside at least one chunk instead of getting cut
# in half between two chunks (which would make it unsearchable).

def split_oversized_block(block: str, chunk_size: int) -> list:
    """
    Guarantee no single block exceeds chunk_size words.
    Tries sentence boundaries first (keeps text more readable),
    falls back to raw word slicing if a single "sentence" is still huge.

    WHY THIS FUNCTION EXISTS (the bug we fixed):
    Our first attempt split documents into "paragraphs" by looking for
    blank lines (two newlines in a row). But pdfplumber usually does NOT
    insert blank lines between paragraphs — it just gives one long stream
    of text with single line breaks. That meant our "paragraph split"
    often found ZERO paragraph breaks and treated an entire PDF as one
    giant block, so no real chunking ever happened. This function is the
    safety net: no matter what the paragraph split gives us, nothing
    bigger than `chunk_size` words is allowed to pass through it.
    """
    # .split() with no arguments splits on any whitespace (spaces, tabs,
    # newlines) and also throws away empty strings from extra spaces —
    # it's the simplest reliable way to count "words" here.
    words = block.split()

    # If this block already fits inside our size limit, there's nothing
    # to do — hand it back unchanged, wrapped in a list (because this
    # function always returns a LIST of blocks, even if that list has
    # just one item in it, so the calling code doesn't need two different
    # code paths for "it fit" vs "it didn't fit").
    if len(words) <= chunk_size:
        return [block]

    # The block is too big — try to break it at sentence boundaries first,
    # since that keeps each piece readable (a chunk that starts or ends
    # mid-sentence is confusing for both search and the LLM later).
    #
    # Pattern breakdown: r"(?<=[.!?])\s+"
    #   (?<=[.!?])  -> a "lookbehind": this checks what comes BEFORE the
    #                  split point, without consuming/removing it. It says
    #                  "only split here if the character right before this
    #                  point is a period, exclamation mark, or question
    #                  mark."
    #   \s+         -> one or more whitespace characters (the space between
    #                  sentences).
    # Together: "split right after a sentence-ending punctuation mark,
    # at the whitespace that follows it." This is what actually gets
    # removed/split on; the punctuation itself stays attached to the
    # sentence before it, which is what we want.
    sentences = re.split(r"(?<=[.!?])\s+", block)
    # Strip whitespace off each piece and drop any that are now empty
    # (can happen if there were multiple spaces/newlines in a row).
    sentences = [s.strip() for s in sentences if s.strip()]

    sub_blocks = []       # the pieces we'll return
    current = []          # sentences we're accumulating into the current piece
    current_len = 0       # running word count of `current`, so we don't
                           # have to re-count it from scratch every loop

    for s in sentences:
        s_len = len(s.split())

        # Edge case: what if a single "sentence" (maybe a run-on list with
        # no periods) is STILL bigger than chunk_size on its own? Sentence
        # splitting alone can't save us here, so we fall back to brute-force
        # slicing it into fixed-size word groups.
        if s_len > chunk_size:
            # First, flush whatever we'd already been building, so we
            # don't lose it or mix it in with this oversized sentence.
            if current:
                sub_blocks.append(" ".join(current))
                current, current_len = [], 0
            s_words = s.split()
            # range(start, stop, step): walk through the word list in
            # jumps of `chunk_size`, e.g. words[0:400], words[400:800], ...
            for i in range(0, len(s_words), chunk_size):
                sub_blocks.append(" ".join(s_words[i:i + chunk_size]))
            # `continue` skips the rest of this loop iteration and moves
            # to the next sentence — we've already fully handled this one.
            continue

        # Normal case: would adding this sentence push us over the limit?
        # `and current` guards against flushing an EMPTY current block
        # (which could otherwise happen on the very first sentence).
        if current_len + s_len > chunk_size and current:
            sub_blocks.append(" ".join(current))
            current, current_len = [], 0

        # Either it fit, or we just flushed and started fresh — add it.
        current.append(s)
        current_len += s_len

    # After the loop ends, whatever's left in `current` hasn't been
    # flushed yet (there's no more sentences to trigger a flush), so we
    # add it as the final piece — but only if it's non-empty.
    if current:
        sub_blocks.append(" ".join(current))

    return sub_blocks


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """
    Chunk by paragraph/section first (better for FAQ/guide docs),
    then merge small paragraphs up to chunk_size, with overlap between chunks.
    Sizes are approximate word counts, not tokens (good enough for v1).
    """
    # --- Pass 1: try to split on real paragraph breaks ---
    # Pattern breakdown: r"\n\s*\n"
    #   \n     -> a newline
    #   \s*    -> zero or more whitespace characters (covers cases where
    #             there's a blank line with trailing spaces on it)
    #   \n     -> another newline
    # In plain English: "a blank line" (two newlines with maybe some
    # whitespace between them). This is how you'd normally detect a
    # paragraph break in clean text.
    raw_blocks = re.split(r"\n\s*\n", text)
    raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

    # --- Pass 2 (the fix): guarantee every block is chunk_size or smaller ---
    # If Pass 1 found real paragraph breaks, this just double-checks none
    # of them are oversized. If Pass 1 found NO breaks at all (the bug we
    # hit — pdfplumber gave us one giant block), this is what actually
    # does the real splitting work, via split_oversized_block().
    atomic_blocks = []
    for block in raw_blocks:
        # .extend() adds every item from the returned list individually
        # (as opposed to .append(), which would nest the whole list inside
        # atomic_blocks as one item — we don't want that here).
        atomic_blocks.extend(split_oversized_block(block, chunk_size))

    # --- Pass 3: greedily merge small blocks up to chunk_size, with overlap ---
    # Now that every block is guaranteed small enough, we walk through
    # them and glue neighboring blocks together until adding one more
    # would push us over chunk_size — that's when we "seal" a chunk and
    # start the next one.
    chunks = []
    current = []      # blocks being accumulated into the chunk we're building
    current_len = 0   # running word count of `current`

    for block in atomic_blocks:
        block_len = len(block.split())

        # Would adding this block overflow the current chunk?
        if current_len + block_len > chunk_size and current:
            # Seal off the chunk we've built so far.
            chunks.append(" ".join(current))

            # OVERLAP LOGIC: take the last `overlap` words of the chunk we
            # just sealed, and use them as the STARTING point of the next
            # chunk. list[-overlap:] means "the last `overlap` items" —
            # e.g. if overlap=80, this grabs the final 80 words.
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)]
            current_len = len(overlap_words)

        # Add the current block (whether we just reset `current` above,
        # or it still had room from before).
        current.append(block)
        current_len += block_len

    # The very last chunk being built never triggers the "overflow" check
    # above (there's no next block to trigger it), so we add it manually
    # here once the loop finishes.
    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_all_docs(docs: dict, chunk_size: int = 400, overlap: int = 80) -> list:
    """Returns list of dicts: {id, text, source, chunk_index}."""
    all_chunks = []

    # docs.items() gives us (filename, text) pairs from the dict built in
    # load_all_pdfs, one pair per PDF.
    for fname, text in docs.items():
        chunks = chunk_text(text, chunk_size, overlap)

        # enumerate(chunks) gives us both the index (0, 1, 2, ...) and the
        # chunk itself, so we can record "this is chunk #2 from this file."
        for i, c in enumerate(chunks):
            all_chunks.append({
                # uuid.uuid4() generates a random unique identifier, e.g.
                # "3f2504e0-4f89-11d3-9a0c-0305e82c3301" — chromadb needs
                # every stored item to have a unique "id" so it doesn't
                # overwrite one chunk with another. str(...) converts the
                # UUID object into a plain string.
                "id": str(uuid.uuid4()),
                "text": c,
                # Keeping the source filename lets us later tell the user
                # exactly which document an answer's information came from
                # (this is the "cite sources" step, coming up in step 7).
                "source": fname,
                "chunk_index": i,
            })
        print(f"{fname}: {len(chunks)} chunks")
    return all_chunks


# ==========================================================================
# STEP 3 & 4: EMBED + STORE — turn text into vectors, save them to disk
# ==========================================================================
#
# WHAT IS AN "EMBEDDING"?
# An embedding model reads a piece of text and outputs a list of numbers
# (a "vector") that represents its MEANING — not its exact words. Text with
# similar meaning ends up with similar-looking vectors, even if the wording
# is completely different (e.g. "get my money back" and "request a refund"
# would land close together). That's what makes semantic search possible:
# instead of matching exact keywords, we compare these number-lists to find
# the closest meaning.

def embed_and_store(chunks: list, db_dir: str, collection_name: str = "taskflow_docs"):
    """
    Embeds chunks with a local sentence-transformers model
    and stores them in a persistent Chroma collection.
    """
    # This sets up a function that knows how to convert text -> vector,
    # using the "all-MiniLM-L6-v2" model. This model runs entirely on your
    # own machine (no API key, no internet needed after the first
    # download, no cost) — that's what "local/free" means in your project
    # notes. Chroma will call this function automatically every time we
    # add or search data, so we never have to call it ourselves directly.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # PersistentClient means the database is saved to an actual folder on
    # disk (db_dir) rather than living only in memory and disappearing
    # when the script ends. This is what lets test_query.py (a totally
    # separate script run later) open the same database and search it.
    client = chromadb.PersistentClient(path=db_dir)

    # A "collection" is like a table in a normal database — a named group
    # of related vectors. We delete any existing collection with this name
    # first so that re-running ingest.py always starts fresh instead of
    # piling up duplicate chunks from previous runs. The try/except is
    # there because delete_collection() raises an error if the collection
    # doesn't exist yet (e.g. on the very first run) — we just want to
    # silently ignore that specific case and move on.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create a brand-new, empty collection, telling it which embedding
    # function to use for anything added to it from now on.
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )

    # This is the actual "embed + store" step happening in one call:
    # collection.add() takes our raw text, runs it through embed_fn behind
    # the scenes to get vectors, and saves the vectors + text + metadata
    # together to disk.
    collection.add(
        # A list of the unique ID we generated for each chunk.
        ids=[c["id"] for c in chunks],
        # The actual chunk text — this is what gets embedded into a vector,
        # and also what we get back later when we search (so we can show
        # the matched text to a user or feed it to an LLM).
        documents=[c["text"] for c in chunks],
        # "Metadata" is extra info stored ALONGSIDE each vector, not used
        # for the similarity math itself, but returned with search results
        # so we know which file (and which chunk number within it) a
        # result came from — this is what powers source citations later.
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
    )

    print(f"Stored {len(chunks)} chunks in collection '{collection_name}' at {db_dir}")
    return collection


# ==========================================================================
# MAIN — ties all the steps together, driven by command-line arguments
# ==========================================================================

def main():
    # argparse lets you run, e.g.:
    #   python ingest.py --docs_dir ./docs --db_dir ./chroma_db --chunk_size 500
    # instead of editing the script every time you want different settings.
    parser = argparse.ArgumentParser()
    # `default=` means "use this value if the user doesn't pass this flag."
    parser.add_argument("--docs_dir", default="./docs")
    parser.add_argument("--db_dir", default="./chroma_db")
    # type=int is important — command-line arguments arrive as plain text
    # by default (e.g. "400" the string, not 400 the number). Without
    # type=int, chunk_size would be a string and break the word-count math.
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    # This actually reads sys.argv (what you typed after "python ingest.py")
    # and turns it into an object where args.docs_dir, args.chunk_size etc.
    # hold the values you passed (or the defaults above).
    args = parser.parse_args()

    print("=== STEP 1: Loading PDFs ===")
    docs = load_all_pdfs(args.docs_dir)

    print("\n=== STEP 2: Chunking ===")
    chunks = chunk_all_docs(docs, args.chunk_size, args.overlap)

    print("\n=== STEP 3 & 4: Embedding + Storing ===")
    embed_and_store(chunks, args.db_dir)

    print("\nDone. Run a quick test query with test_query.py")


# This check means "only run main() if this file was executed directly
# (python ingest.py), NOT if some other script does `import ingest` to
# reuse its functions." It's a standard Python convention that keeps
# scripts reusable as importable modules without them auto-running.
if __name__ == "__main__":
    main()