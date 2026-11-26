from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from google import genai

from jansahayak_agent import JanSahayakAgent

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "JanSahayak"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(
    page_title="JanSahayak | Uttarakhand Government Service Navigator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main .block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
      .hero {padding: 1.1rem 1.3rem; border: 1px solid rgba(120,120,120,.25); border-radius: 16px;}
      .muted {color: #888; font-size: .88rem;}
      a {overflow-wrap: anywhere;}
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_traces" not in st.session_state:
    st.session_state.research_traces = {}

with st.sidebar:
    st.title("🏛️ JanSahayak")
    st.caption("Dynamic government-service discovery and application guidance for Uttarakhand.")
    st.divider()
    st.markdown("**Research mode**")
    st.success("Live web discovery enabled")
    st.caption("Uses the DDGS metasearch library (no separate search API key). Results may be rate-limited or temporarily unavailable.")
    st.caption(f"Gemini model: `{model}`")
    st.divider()
    location = st.text_input(
        "Your district / town (optional)",
        placeholder="e.g. Roorkee, Haridwar, Dehradun",
        help="Useful for municipal complaints and district-specific services. You can also include your location in the question.",
    )
    st.divider()
    st.markdown("**Try asking**")
    st.caption("I need to complain about garbage collection in Roorkee.")
    st.caption("How do I apply for a domicile certificate in Haridwar?")
    st.caption("My electricity complaint has not been resolved. What can I do?")
    st.caption("I need to apply for an old-age pension for my mother.")
    st.caption("Where can I report a problem with a government ration shop?")
    st.divider()
    st.markdown("**Important safeguards**")
    st.caption("JanSahayak searches for public government information. It does not submit forms or payments.")
    st.caption("Never enter Aadhaar numbers, OTPs, passwords, bank details or private document scans.")
    st.caption("Only HTTPS .gov.in and .nic.in pages are presented as official sources. Some legitimate sites outside these domains may be excluded for manual review.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.research_traces = {}
        st.rerun()

st.title("🏛️ JanSahayak")
st.subheader("Uttarakhand Government Service Navigator")
st.markdown(
    "Describe your problem in ordinary language. JanSahayak researches the live web, "
    "finds relevant government authorities, reads official pages and explains the next steps. "
    "It is not restricted to a predefined service catalogue or a fixed number of services."
)

with st.expander("How JanSahayak researches your request", expanded=False):
    st.markdown(
        "1. Understands your intent and identifies missing context.\n"
        "2. Plans targeted searches for Uttarakhand and applicable Central Government sources.\n"
        "3. Searches the live web and filters results to HTTPS `.gov.in` / `.nic.in` domains.\n"
        "4. Opens official pages, extracts readable text and follows relevant official links.\n"
        "5. Uses Gemini to prepare a guide grounded in the retrieved evidence.\n\n"
        "There is no fixed service catalogue or 12-service limit. Each request is researched dynamically. The current run searches up to 8 queries, retrieves up to 8 primary pages and follows up to 8 relevant official links."
    )

if not api_key:
    st.warning("Gemini is not configured. Add GEMINI_API_KEY to the .env file in this project folder, then restart Streamlit.")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        trace = st.session_state.research_traces.get(str(index))
        if trace and message["role"] == "assistant":
            with st.expander("Research details and official sources"):
                if trace.get("queries"):
                    st.markdown("**Search queries used**")
                    for query in trace["queries"]:
                        st.write(f"- {query}")
                if trace.get("sources"):
                    st.markdown("**Official-domain pages discovered**")
                    for source in trace["sources"]:
                        status = "Page retrieved" if source.get("fetch_status") == "page_retrieved" else "Search result only; page text unavailable"
                        st.markdown(f"- [{source['title']}]({source['url']}) — {status}")
                if trace.get("search_errors"):
                    st.warning("Some search attempts failed or were rate-limited. The research may be incomplete.")
                    for error in trace["search_errors"]:
                        st.caption(error)
                if not trace.get("sources"):
                    st.warning("No official-domain search results were found for this request.")

prompt = st.chat_input("What government service or problem do you need help with?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            answer = "JanSahayak is not configured. Add GEMINI_API_KEY to the local .env file and restart the app."
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            try:
                client = genai.Client(api_key=api_key)
                agent = JanSahayakAgent(client=client, model=model)
                status = st.status("Starting JanSahayak research…", expanded=True)

                def update_status(message: str) -> None:
                    status.update(label=message, state="running")
                    status.write(message)

                history = st.session_state.messages[:-1][-8:]
                result = agent.answer(
                    user_query=prompt,
                    history=history,
                    location=location,
                    progress=update_status,
                )
                status.update(label="Research completed", state="complete", expanded=False)
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(f"Research details and official sources ({len(result['sources'])})"):
                        st.markdown("**Search queries used**")
                        for query in result.get("queries", []):
                            st.write(f"- {query}")
                        st.markdown("**Official-domain pages discovered**")
                        for source in result["sources"]:
                            status_text = "Page retrieved" if source.get("fetch_status") == "page_retrieved" else "Search result only; page text unavailable"
                            st.markdown(f"- [{source['title']}]({source['url']}) — {status_text}")
                else:
                    st.warning("The live search did not return any HTTPS .gov.in or .nic.in results. Try adding your town/district or rephrasing the service name. No application URL has been invented.")
                if result.get("search_errors"):
                    st.info("Some search providers may have rate-limited the request. Treat the research as incomplete and retry later.")

                assistant_index = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.session_state.research_traces[str(assistant_index)] = {
                    "queries": result.get("queries", []),
                    "sources": result.get("sources", []),
                    "search_errors": result.get("search_errors", []),
                }
            except Exception as exc:
                error = f"JanSahayak encountered an error: {type(exc).__name__}: {exc}"
                st.error(error)
                st.caption("Check your Gemini API key, model name, internet connection and installed packages. Do not share your API key in screenshots.")
                st.session_state.messages.append({"role": "assistant", "content": error})
