# UniGuide-RAG

**An evidence-grounded demo campus information assistant using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs).**

UniGuide-RAG is a self-contained academic prototype. It ships with a small, fully fictional sample institution named **Northstar Demo University** and synthetic PDF documents so the application can be demonstrated without using any real college or university data. The fictional institution, academic rules, dates, services and contact details in the sample PDFs are invented for software testing and are not real policies.

The application also allows users to index their own public or non-confidential PDF documents. It extracts page-aware passages, retrieves relevant text, calls Gemini to draft an answer from the retrieved evidence, and displays the supporting source and page. If Gemini embeddings are unavailable, retrieval falls back to TF-IDF. If no Gemini API key is configured, an extractive retrieval-only mode remains available.

> **Important:** This is not an official university service and is not affiliated with any real institution. Do not use the fictional demo policies for actual academic decisions. Do not upload confidential student records, personal data or documents you are not authorised to process.

## 1. Features

- Built-in synthetic demo corpus: student handbook, assessment and examination guide, academic calendar, and library/IT services guide.
- One-click loading of the demo PDFs from the Knowledge Base tab.
- PDF ingestion using PyMuPDF.
- Page-aware overlapping text chunks.
- Gemini embedding retrieval when configured.
- TF-IDF cosine-similarity fallback.
- Gemini LLM answer generation with evidence-only instructions.
- Source tags and retrieved passage inspection.
- Conservative abstention when retrieved evidence is inadequate.
- Local JSON knowledge-base storage.
- Retrieval benchmark and HitRate@k / MRR@k evaluation script.
- Unit tests for chunking, PDF extraction and retrieval.

## 2. Requirements

- Python 3.10 or later (Python 3.11 or 3.12 recommended).
- Internet connection for installing dependencies and Gemini API requests.
- Optional Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 3. Installation (Windows PowerShell)

Open PowerShell in the extracted `UniGuide-RAG` folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

For LLM-generated answers, paste your Gemini API key after `GEMINI_API_KEY=` in `.env`. Do not share or upload `.env` to GitHub. The application can still demonstrate PDF ingestion and TF-IDF retrieval without a key; it will clearly label the response as retrieval-only rather than pretending an LLM ran. The built-in lexical relevance threshold is tuned for this small synthetic corpus and should be recalibrated for a different dataset or dense-embedding configuration.

The default model identifiers are configured in `.env.example` and can be changed to models available to your API key.

## 4. Run the application

```powershell
streamlit run app.py
```

In the browser:

1. Open **Knowledge Base**.
2. Click **Load built-in demo documents** to load the fictional Northstar Demo University corpus.
3. Open **Ask UniGuide** and ask one of the sample questions below.
4. Expand **Retrieved evidence** to inspect the exact document and page used.

You may alternatively upload other text-searchable PDFs, provided you have permission and they contain no confidential personal data.

## 5. Built-in demo questions

- What is the minimum attendance requirement in the demo handbook?
- How many working days are available to request a review of an assessment mark?
- When does the demo autumn semester begin?
- How many books may a student borrow at one time?
- What are the library's Saturday opening hours?
- How can a student request a password reset?
- Does the demo document specify a hostel fee?

The last question is intentionally unanswerable from the supplied demo corpus and should trigger a cautious response.

## 6. Evaluate retrieval

The benchmark CSV contains labelled questions based on the built-in synthetic PDFs. After loading the demo corpus at least once, run:

```powershell
python evaluation/evaluate_retrieval.py --k 5
```

The script reports source-level HitRate@k and MRR@k. These metrics evaluate whether the expected document appears among retrieved results; they do not measure answer correctness. Add reviewed questions and labels before making research claims.

Run unit tests:

```powershell
python -m pytest -q
```

## 7. Project structure

```text
UniGuide-RAG/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── sample_data/                 # fictional demo PDFs
├── uniguide/                    # ingestion, retrieval, generation, storage
├── data/                        # local knowledge base created at runtime
├── evaluation/                  # labelled retrieval benchmark and evaluator
├── tests/                       # automated tests
└── research/                    # paper draft and architecture diagram
```

## 8. Research evaluation plan

The paper draft is in `research/UniGuide-RAG_Research_Paper_Draft.md`. It distinguishes the synthetic demonstration corpus from real institutional data. Recommended experiments include TF-IDF versus Gemini embeddings, HitRate@5 and MRR@5, answer correctness, citation support, abstention behaviour, response latency and ablation tests. Results must be measured using a documented benchmark; do not invent accuracy values or claim that results on synthetic policies generalise to real universities.

## 9. Security and responsible use

- Keep API keys in `.env` or Streamlit secrets; never hard-code them.
- Do not upload private student records, marksheets, IDs, medical information or personal data.
- Upload only documents you are authorised to process.
- Treat generated answers as assistance, not official rulings.
- Verify real policies against current authoritative documents.

## 10. Current limitations

This is an academic prototype, not a production-ready campus deployment. It uses a local JSON store, has no authentication or multi-user access controls, does not perform OCR, does not automatically authenticate a document or confirm that it is the latest version, and has only a small synthetic demonstration corpus. These limitations must be acknowledged in the research paper.
