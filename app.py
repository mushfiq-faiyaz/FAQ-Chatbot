"""
RAG Chat UI — Step 8: Simple Chat UI
(Updated: error handling around the Groq call + a "Clear chat" button)

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
Every previous script (ingest.py, test_query.py, rag_chat.py) only ran in
a terminal, one question at a time. This file wraps rag_chat.py's
answer_question() function in an actual clickable chat window that opens
in your browser — the same "type a message, see a reply appear" pattern
you're used to from any chat app.

We are NOT rebuilding retrieval or generation here. Every bit of "brain"
logic (retrieve chunks, call Groq, format citations) already lives in
rag_chat.py. This file's only job is the INTERFACE: showing messages,
remembering the conversation, and calling into that existing logic.

WHAT CHANGED IN THIS VERSION (and why):
1. The call to answer_question() is now wrapped in try/except. Before,
   if Groq raised an error (bad key, rate limit, network hiccup, model
   deprecated, etc.), the crash happened AFTER we'd already saved the
   user's question into st.session_state.messages but BEFORE we saved
   any assistant reply. That left an "orphaned" question sitting in the
   chat forever with no answer — which is exactly the double-question
   bug you saw in your screenshot. Now, any failure gets caught and
   turned into a normal-looking assistant message ("Sorry, something
   went wrong...") that gets saved just like a real answer would. The
   conversation history stays consistent no matter what happens.
2. A "Clear chat" button in the sidebar resets st.session_state.messages
   back to an empty list, so you (or someone you're demoing this to) can
   start a fresh conversation without restarting the whole app.

Usage:
    pip install streamlit --break-system-packages

    Run it with the special streamlit command (NOT "python app.py" —
    see the note at the bottom of this file for why):
        streamlit run app.py

    This opens a browser tab automatically at http://localhost:8501
"""

# streamlit is imported as `st` by convention everywhere in Streamlit's
# own docs and examples — you'll see `st.` in front of almost everything
# in this file, since nearly every line is asking Streamlit to draw or
# remember something.
import streamlit as st

# WHY THIS IMPORT IS HERE:
# rag_chat.py only calls load_dotenv() inside its own main() function,
# which only runs when you execute "python rag_chat.py" directly. When
# app.py imports answer_question() from rag_chat.py, it does NOT trigger
# that main() function — it just grabs the function definition. That
# means .env never gets loaded through that path, and GROQ_API_KEY would
# be missing when Groq() tries to read it. So we load it here too,
# independently, since app.py is its own separate entry point into the
# program (started via "streamlit run app.py", not "python rag_chat.py").
from dotenv import load_dotenv

from rag_chat import answer_question

# This must run BEFORE any code that needs GROQ_API_KEY — since
# everything in this file runs top-to-bottom on every re-run, putting it
# here at the very top guarantees the key is loaded before st.chat_input
# further down ever has a chance to trigger a Groq() call.
load_dotenv()


# ==========================================================================
# PAGE SETUP — runs once per script load, configures the browser tab
# ==========================================================================

# st.set_page_config MUST be the first Streamlit command in the file (a
# Streamlit rule, not a style choice) — it sets the browser tab's title
# and icon before anything else gets drawn.
st.set_page_config(page_title="TaskFlow Support Chat", page_icon="💬")

# Renders a large heading at the top of the page. This is plain Streamlit
# syntax: st.<something>(...) draws that "something" on the page, in the
# order you call it in the script — top to bottom, just like reading it.
st.title("💬 TaskFlow Support Chat")
st.caption("Ask a question about TaskFlow — answers are grounded in the official docs.")


# ==========================================================================
# THE CORE PROBLEM: STREAMLIT RE-RUNS THE WHOLE SCRIPT ON EVERY INTERACTION
# ==========================================================================
#
# This is the single most important thing to understand about Streamlit,
# and it's different from how a normal Python script behaves:
#
# Every time you type a message and hit enter, Streamlit does NOT just
# run "the new part" — it re-runs THIS ENTIRE FILE FROM THE TOP, as if
# you'd freshly typed `streamlit run app.py` again. Any plain Python
# variable (like `messages = []`) would get wiped out and recreated as
# an empty list on every single re-run — you'd lose the whole
# conversation after every message.
#
# st.session_state solves this: it's a special dictionary-like object
# that Streamlit keeps ALIVE across re-runs, tied to your specific
# browser session. As long as you don't close the tab, whatever you
# store in it survives.

# "if key not in session_state" is the standard Streamlit pattern for
# "only initialize this ONCE." Without this check, every re-run would
# reset `messages` back to an empty list right before we try to display
# the history we just built — wiping the conversation on every message.
if "messages" not in st.session_state:
    # We're storing the conversation as a list of small dicts, e.g.:
    #   {"role": "user", "content": "How do I get a refund?"}
    #   {"role": "assistant", "content": "...", "sources": "..."}
    # "role" tracks WHO said it, so we can style user vs. bot messages
    # differently when we redraw them further down.
    st.session_state.messages = []


# ==========================================================================
# SIDEBAR — "Clear chat" button
# ==========================================================================
#
# st.sidebar puts whatever you draw on it into the collapsible sidebar
# panel instead of the main chat column — a natural home for controls
# that AREN'T part of the conversation itself (settings, resets, etc.),
# so they don't clutter the chat bubbles.
with st.sidebar:
    st.header("Controls")
    # st.button() returns True on the exact re-run where it was clicked,
    # and False on every other re-run — same "only true once" pattern as
    # st.chat_input() further down.
    if st.button("🗑️ Clear chat"):
        # Resetting the list back to empty is all it takes — the redraw
        # loop below reads from this same list, so an empty list means
        # nothing gets drawn, i.e. a fresh-looking chat.
        st.session_state.messages = []
        # st.rerun() immediately restarts the script from the top instead
        # of waiting for the next natural interaction. Without this, the
        # OLD messages would still be visible on screen until you did
        # something else to trigger a re-run (like sending a message) —
        # the click itself wouldn't visibly clear anything right away.
        st.rerun()


# ==========================================================================
# REDRAW EVERY PAST MESSAGE
# ==========================================================================
#
# Because the whole script re-runs on every interaction, we have to
# manually redraw the ENTIRE conversation history every single time —
# Streamlit doesn't remember what was drawn on screen before, only what's
# in session_state. This loop is what makes old messages stay visible
# instead of disappearing every time you send a new one.

for msg in st.session_state.messages:
    # st.chat_message(role) is a special Streamlit component that draws a
    # chat bubble styled for that role (e.g. a bot icon for "assistant",
    # a person icon for "user") — it's built specifically for this exact
    # chat-UI use case.
    #
    # The "with" block means everything indented under it gets drawn
    # INSIDE that chat bubble, not just anywhere on the page.
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Only assistant messages carry a "sources" key (we never attach
        # sources to what the user typed) — .get() returns None instead
        # of crashing if the key doesn't exist on a given message.
        if msg.get("sources"):
            # st.caption renders small, muted/gray text — a natural fit
            # for a "Sources:" footer that shouldn't compete visually
            # with the actual answer above it.
            st.caption(f"Sources:\n{msg['sources']}")


# ==========================================================================
# HANDLE A NEW MESSAGE
# ==========================================================================

# st.chat_input draws the text box fixed at the bottom of the screen
# (the same layout you'd expect from any chat app) and returns whatever
# the user typed, but ONLY on the run where they actually hit enter —
# on every other re-run it returns None. That's why everything below is
# wrapped in "if prompt:" — None is falsy in Python, so this block is
# skipped entirely except on the exact run where a new message arrives.
prompt = st.chat_input("Ask a question about TaskFlow...")

if prompt:
    # Step 1: save the user's new message into history immediately, and
    # draw it right away — otherwise it wouldn't appear until the NEXT
    # re-run, which would feel laggy and confusing.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 2: generate the bot's reply. This block is where we finally
    # call into rag_chat.py's logic — retrieve, generate, cite — all in
    # one function call, exactly like our CLI script did.
    with st.chat_message("assistant"):
        # st.spinner shows a small animated "loading" indicator with the
        # given text while the code inside the "with" block runs — useful
        # here because the Groq API call can take a second or two, and
        # without this the page would just look frozen with no feedback.
        with st.spinner("Searching TaskFlow docs..."):
            # THE FIX: everything that can fail (retrieval, the Groq call,
            # anything inside answer_question) now happens inside a try
            # block. `except Exception as e` catches ANY kind of error —
            # deliberately broad here, because from the UI's perspective
            # it doesn't matter WHY the call failed (bad key, network
            # drop, rate limit, Groq's servers being down); the response
            # to the user should look the same either way: a clear,
            # honest error bubble instead of the app silently breaking.
            try:
                result = answer_question(prompt, db_dir="./chroma_db")
                answer_text = result["answer"]
                sources_text = result["sources"]
            except Exception as e:
                # We show a friendly message to the user, but we still
                # print the real exception to the terminal running
                # `streamlit run app.py` — that's where YOU (the
                # developer) can see the actual error type/message for
                # debugging, without exposing raw stack traces to
                # whoever is using the chat.
                print(f"[ERROR] answer_question failed: {e}")
                answer_text = (
                    "Sorry, I ran into a problem answering that just now. "
                    "Please try again in a moment."
                )
                # No sources to show for a failed request — leaving this
                # as an empty string keeps the `if msg.get("sources")`
                # check further up (and the one just below) working
                # correctly, since empty strings are falsy in Python too.
                sources_text = ""

        st.markdown(answer_text)
        if sources_text:
            st.caption(f"Sources:\n{sources_text}")

    # Step 3: save the bot's reply into history too, so it survives the
    # next re-run (i.e. it stays on screen after you send another
    # message). Without this line, the answer would display once and
    # then vanish the moment you typed anything new.
    #
    # NOTE: this now ALWAYS runs, whether the try block above succeeded
    # or hit the except — either way we have an answer_text (real or the
    # friendly error message) and a sources_text (real or empty) ready to
    # save. This is exactly what fixes the orphaned-message bug: the
    # user's question and SOME assistant reply are now always saved as a
    # matched pair, never one without the other.
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_text,
    })


# ==========================================================================
# WHY "streamlit run app.py" INSTEAD OF "python app.py"?
# ==========================================================================
# A normal Python script runs once, top to bottom, and exits. A web app
# needs to: start a local web server, keep listening for you to open the
# browser tab, and re-run this script's code every time you interact with
# it. The `streamlit` command wraps your plain script with all of that
# server/re-run machinery — "python app.py" would just run this file once
# with no server behind it and immediately exit, which is why Streamlit
# apps are always launched with `streamlit run <file>`, never `python
# <file>`.