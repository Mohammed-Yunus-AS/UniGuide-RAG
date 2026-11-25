from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from uniguide.config import settings
from uniguide.generation import extractive_fallback, generate_answer, make_client
from uniguide.ingestion import extract_pdf_chunks, validate_uploads
from uniguide.retrieval import Retriever, source_label
from uniguide.storage import load_chunks, remove_sources, save_chunks

st.set_page_config(page_title="UniGuide-RAG | Demo Campus Assistant", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1180px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#173b63,#286b82); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0 0 .25rem 0;}
.hero p {color: #e8f2f7; margin: 0;}
.small-note {color: #667085; font-size: .88rem;}
.source-card {padding: .75rem 1rem; border: 1px solid #d9e2ec; border-radius: 10px; margin: .45rem 0;}
</style>
<div class="hero"><h1>🎓 UniGuide-RAG</h1><p>A demo campus information assistant powered by Retrieval-Augmented Generation and LLMs.</p></div>
""", unsafe_allow_html=True)

# API key may be supplied through .env or Streamlit secrets.
api_key = settings.gemini_api_key
try:
    api_key = st.secrets.get("GEMINI_API_KEY", api_key)
except Exception:
    pass

@st.cache_resource(show_spinner=False)
def get_gemini_client(key: str | None):
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)

client = get_gemini_client(api_key)

if "chunks" not in st.session_state:
    st.session_state.chunks = load_chunks()
if "messages" not in st.session_state:
    st.session_state.messages = []

chunks = st.session_state.chunks
sources = sorted({c.get("source", "") for c in chunks if c.get("source")})

with st.sidebar:
    st.header("Demo Project Status")
    st.metric("Indexed text chunks", len(chunks))
    st.metric("Documents", len(sources))
    st.success("Gemini LLM connected") if client else st.warning("Gemini key not configured")
    st.caption(f"Generation model: {settings.gemini_model}")
    st.caption("Retrieval: Gemini embeddings when available; TF-IDF fallback otherwise.")
    st.divider()
    st.subheader("Quick setup")
    st.markdown("1. Optionally add your Gemini key to `.env`.\n2. In **Knowledge Base**, click **Load built-in demo documents**.\n3. Or upload your own non-confidential sample PDFs.\n4. Ask a question and inspect its sources.")
    st.caption("The built-in Northstar Demo University is fictional. Do not upload confidential student records or personal data.")

ask_tab, docs_tab, guide_tab = st.tabs(["💬 Ask UniGuide", "📚 Knowledge Base", "ℹ️ About & Method"])

with docs_tab:
    st.subheader("Demo knowledge base")
    st.write("Start with the fictional Northstar Demo University documents included in this project, or upload your own public/non-confidential sample PDFs.")
    sample_dir = Path(__file__).parent / "sample_data"
    sample_files = sorted(sample_dir.glob("*.pdf")) if sample_dir.exists() else []
    if sample_files:
        st.info("Built-in dataset: synthetic documents for **Northstar Demo University**. This institution and every rule in its documents are fictional and are included only for software demonstration and testing.")
        if st.button("📦 Load built-in demo documents", type="primary", key="load_demo_docs"):
            demo_chunks = []
            demo_errors = []
            for sample_path in sample_files:
                try:
                    demo_chunks.extend(extract_pdf_chunks(sample_path.read_bytes(), sample_path.name))
                except Exception as exc:
                    demo_errors.append(f"{sample_path.name}: {exc}")
            if demo_errors:
                for error in demo_errors:
                    st.error(error)
            if demo_chunks:
                # Loading the built-in demo corpus resets any previously indexed sources
                # so that the demo contains only the synthetic sample institution.
                merged = demo_chunks
                with st.spinner("Preparing demo retrieval index…"):
                    retriever_for_embedding = Retriever(merged, client=client)
                    _, message = retriever_for_embedding.add_embeddings(merged)
                save_chunks(merged)
                st.session_state.chunks = merged
                st.session_state.messages = []
                st.success(f"Loaded {len(demo_chunks)} demo text chunks from {len(sample_files)} fictional documents. {message}")
                st.rerun()
    uploaded_files = st.file_uploader("Choose PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_uploads")
    if uploaded_files:
        errors = validate_uploads(uploaded_files)
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.info(f"{len(uploaded_files)} PDF file(s) ready for indexing. Each file is limited to 25 MB.")
    if st.button("⚙️ Index documents", type="primary", disabled=not uploaded_files):
        errors = validate_uploads(uploaded_files)
        if errors:
            st.error("Please correct the upload errors first.")
        else:
            new_chunks = []
            progress = st.progress(0)
            for i, uploaded in enumerate(uploaded_files, start=1):
                try:
                    new_chunks.extend(extract_pdf_chunks(uploaded.getvalue(), uploaded.name))
                except Exception as exc:
                    st.error(str(exc))
                progress.progress(i / len(uploaded_files))
            if new_chunks:
                replace_names = {file.name for file in uploaded_files}
                merged = remove_sources(load_chunks(), replace_names) + new_chunks
                with st.spinner("Preparing retrieval index…"):
                    retriever_for_embedding = Retriever(merged, client=client)
                    count, message = retriever_for_embedding.add_embeddings(merged)
                save_chunks(merged)
                st.session_state.chunks = merged
                st.success(f"Indexed {len(new_chunks)} text chunks from the selected PDFs. {message}")
                st.rerun()

    st.divider()
    st.subheader("Indexed documents")
    current_chunks = st.session_state.chunks
    current_sources = sorted({c.get("source", "") for c in current_chunks if c.get("source")})
    if not current_sources:
        st.info("No documents indexed yet. Click **Load built-in demo documents** or upload sample PDFs to build the knowledge base.")
    else:
        for source in current_sources:
            source_chunks = [c for c in current_chunks if c.get("source") == source]
            pages = sorted({c.get("page") for c in source_chunks})
            has_embeddings = all(isinstance(c.get("embedding"), list) for c in source_chunks)
            col1, col2, col3 = st.columns([4, 2, 2])
            col1.write(f"**{source}**")
            col2.caption(f"{len(source_chunks)} chunks · {len(pages)} pages")
            col3.caption("Gemini vectors" if has_embeddings else "TF-IDF fallback")
        selected_to_remove = st.multiselect("Select documents to remove", current_sources)
        if st.button("Remove selected documents", disabled=not selected_to_remove):
            remaining = remove_sources(current_chunks, set(selected_to_remove))
            save_chunks(remaining)
            st.session_state.chunks = remaining
            st.success("Selected documents removed from the local knowledge base.")
            st.rerun()

with ask_tab:
    st.subheader("Ask the demo campus assistant")
    st.caption("Answers are grounded in the documents indexed in this demo knowledge base. The included Northstar Demo University is fictional; its rules are not real institutional policies.")
    with st.expander("Try sample questions"):
        st.markdown("""
        - What is the minimum attendance requirement?
        - How many working days do students have to request a mark review?
        - When does the autumn semester begin?
        - How many books can a student borrow?
        - What is the hostel fee? (The demo corpus does not contain this information.)
        """)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved evidence"):
                    for idx, item in enumerate(message["sources"], start=1):
                        st.markdown(f"**[S{idx}] {source_label(item)}** · relevance {item.get('score', 0):.3f}")
                        st.write(item.get("text", ""))
    question = st.chat_input("Example: How many days do students have to request an exam-mark review?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if not st.session_state.chunks:
                answer = "The demo knowledge base is empty. Open **Knowledge Base** and load the built-in fictional demo documents, or upload sample PDFs."
                evidence = []
            else:
                with st.spinner("Searching demo documents and preparing an answer…"):
                    start_time = time.perf_counter()
                    retriever = Retriever(st.session_state.chunks, client=client)
                    evidence = retriever.search(question, top_k=settings.top_k)
                    if client:
                        try:
                            answer = generate_answer(client, question, evidence)
                        except Exception as exc:
                            answer = ("The Gemini request failed. Check the API key, model access and network connection. "
                                      f"Technical detail: {exc}")
                    else:
                        answer = extractive_fallback(evidence)
                    elapsed = time.perf_counter() - start_time
            st.markdown(answer)
            if evidence:
                with st.expander("Retrieved evidence"):
                    for idx, item in enumerate(evidence, start=1):
                        st.markdown(f"**[S{idx}] {source_label(item)}** · relevance {item.get('score', 0):.3f}")
                        st.write(item.get("text", ""))
            st.caption("Response time: measured locally for this request; retrieval scores are not probabilities.")
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": evidence})

with guide_tab:
    st.subheader("How the demo works")
    st.markdown("""
    **1. Document ingestion** — PyMuPDF extracts selectable text from each PDF and retains page-level source metadata.

    **2. Chunking** — Long pages are divided into overlapping text segments so that relevant passages can be retrieved.

    **3. Embedding and retrieval** — When the Gemini embedding API is available, document vectors are stored locally and compared with query vectors. If embeddings are unavailable, the prototype falls back to TF-IDF cosine similarity.

    **4. Grounded generation** — Retrieved passages are sent to Gemini with instructions to answer only from the supplied evidence and to abstain when evidence is insufficient.

    **5. Source inspection** — The interface displays the retrieved document name, page and passage so the student can verify the response.
    """)
    st.warning("Northstar Demo University and its policies are entirely fictional. This academic prototype is not affiliated with any real university and must not be used for actual academic, financial or administrative decisions. PDF scans without selectable text are not processed in this MVP.")
    st.markdown("**Research measurements to collect:** Precision@K, Recall@K, MRR, answer correctness judged against a prepared test set, abstention quality, and end-to-end response time.")
