import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz

from uniguide.ingestion import extract_pdf_chunks, split_into_chunks
from uniguide.retrieval import Retriever


def test_chunking_preserves_content_and_overlap():
    text = "University examination regulation information. " * 60
    chunks = split_into_chunks(text, chunk_size=350, overlap=60)
    assert len(chunks) > 1
    assert all(chunks)
    assert "University" in chunks[0]


def test_extract_page_aware_pdf_chunks():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Official University Examination Policy: Students may apply for revaluation.")
    raw = doc.tobytes()
    doc.close()
    chunks = extract_pdf_chunks(raw, "sample_policy.pdf")
    assert chunks
    assert chunks[0]["source"] == "sample_policy.pdf"
    assert chunks[0]["page"] == 1
    assert "revaluation" in chunks[0]["text"].lower()


def test_tfidf_retriever_returns_relevant_source():
    chunks = [
        {"id": "1", "text": "The examination revaluation application is submitted through the student portal.",
         "source": "exam_policy.pdf", "page": 4},
        {"id": "2", "text": "The library is open from Monday to Friday during semester days.",
         "source": "library_guide.pdf", "page": 2},
    ]
    retriever = Retriever(chunks, client=None)
    results = retriever.search("How do I apply for examination revaluation?", top_k=2)
    assert results[0]["source"] == "exam_policy.pdf"
    assert results[0]["score"] > 0


def test_builtin_demo_pdfs_are_synthetic_and_page_aware():
    sample_dir = Path(__file__).resolve().parents[1] / "sample_data"
    pdfs = sorted(sample_dir.glob("*.pdf"))
    assert len(pdfs) == 4
    all_chunks = []
    for pdf in pdfs:
        extracted = extract_pdf_chunks(pdf.read_bytes(), pdf.name)
        assert extracted
        assert all(item["page"] in (1, 2) for item in extracted)
        combined_text = " ".join(item["text"] for item in extracted).lower()
        assert "northstar" in combined_text or "fictional" in combined_text or "demo" in combined_text
        all_chunks.extend(extracted)

    retriever = Retriever(all_chunks, client=None)
    results = retriever.search("How many working days can a student request a review of an assessment mark?", top_k=5)
    assert results
    assert results[0]["source"] == "northstar_assessment_exam_guide_demo.pdf"
