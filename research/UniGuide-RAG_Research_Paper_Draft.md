# UniGuide-RAG: An Evidence-Grounded Demo Campus Information Assistant Using Retrieval-Augmented Generation and Large Language Models

**Author(s):** [Student name(s)]  
**Affiliation:** [Department / Institution, if required by the course]  
**Corresponding author:** [Email]

> **Dataset note:** Northstar Demo University is fictional. The accompanying PDF corpus and its policies are synthetic test material, not official or real institution information.
>
> **Draft status:** This is an initial research-paper draft and experiment plan. Replace bracketed fields and complete the experiments before presenting it as a final research paper. No experimental results have been fabricated.

## Abstract

Institutional information is often distributed across academic regulations, examination notifications, student handbooks and service guides. Users may find it difficult to locate the correct rule or identify the latest relevant notice, while general-purpose large language models (LLMs) can produce unsupported statements when they are not grounded in institution-style evidence. This paper presents **UniGuide-RAG**, a document-grounded campus information assistant based on Retrieval-Augmented Generation (RAG). The prototype extracts text from synthetic demo PDF documents (or authorised sample PDFs), preserves page-level metadata, divides the content into overlapping chunks, retrieves relevant passages using vector similarity when Gemini embeddings are available, and supplies the retrieved evidence to a Gemini LLM for answer generation. A TF-IDF retrieval fallback is provided for operation without the embedding service. The generation prompt requires evidence-based answers, source tags and abstention when the retrieved passages do not support a response. The research evaluates retrieval quality, answer correctness, source attribution, abstention behaviour and response time using a manually reviewed synthetic demo-question benchmark. The intended contribution is an end-to-end, reproducible demo campus information assistant that combines source traceability, a conservative unsupported-answer policy and document-aware retrieval. Quantitative findings will be reported after the benchmark is populated and the implemented system is tested.

**Keywords:** Generative Artificial Intelligence; Large Language Models; Retrieval-Augmented Generation; Campus Information Assistant; Semantic Retrieval; Evidence Grounding; Hallucination Mitigation.

## 1. Introduction

Universities publish information through multiple document formats and channels, including academic regulations, examination notices, programme handbooks and student service guidelines. Students and other users often need to search these materials to answer practical questions about procedures, eligibility, deadlines and institutional services. Keyword search may fail when a student's wording differs from the wording used in a policy document. Manual enquiry channels can also require users to identify the correct office before they know which document contains the answer.

Large language models provide a natural-language interface for asking questions, summarising information and explaining complex text. However, a general-purpose LLM does not necessarily contain the current rules of a particular institution. A response can sound confident even when supporting institutional evidence is absent. This creates a need for systems that connect natural-language generation to a controlled collection of source documents.

Retrieval-Augmented Generation addresses this problem by retrieving relevant passages from an external knowledge collection and conditioning generation on those passages. UniGuide-RAG applies this design to campus information. It focuses on page-aware document ingestion, evidence retrieval, source-attributed answer generation and abstention when evidence is insufficient.

### 1.1 Problem Statement

Users require a simple way to obtain understandable answers from institution-style documents. Keyword-based search and generic conversational models do not by themselves guarantee that a generated answer is supported by the material in a given document collection. The problem addressed in this work is to design and evaluate a prototype that retrieves relevant document-grounded passages and generates responses that remain traceable to those passages.

### 1.2 Objectives

1. Develop a PDF ingestion pipeline that extracts text and retains filename and page metadata.
2. Implement chunking and retrieval over a local document-grounded knowledge base.
3. Integrate an LLM to generate concise answers conditioned on retrieved evidence.
4. Display source-document and page references alongside retrieved evidence.
5. Implement an abstention response when evidence is missing or below a configurable retrieval threshold.
6. Evaluate retrieval performance, answer quality, source attribution, abstention behaviour and latency.

### 1.3 Contributions

The prototype makes the following implementation-level contributions:

- A page-aware ingestion workflow for institution-style PDF documents.
- A retrieval layer that supports Gemini retrieval embeddings and a TF-IDF fallback.
- An evidence-constrained LLM prompt that discourages unsupported policy claims.
- A source-inspection interface that exposes the retrieved text, document and page.
- A benchmark protocol for evaluating retrieval and answer quality on reviewed demo questions.

These contributions are application- and evaluation-oriented. The use of RAG, vector embeddings and LLMs is established prior work; the paper does not claim to invent these techniques.

## 2. Related Work

### 2.1 Large Language Models

LLMs generate text by modelling patterns in large collections of language data. They can interpret natural-language questions and produce fluent explanations, but fluency should not be treated as proof that a factual response is correct. Institution-specific assistants therefore need a mechanism for connecting generated text to authoritative local information.

### 2.2 Retrieval-Augmented Generation

Lewis et al. introduced a general RAG approach that combines a parametric generation model with a non-parametric retrieved knowledge source. In a typical RAG pipeline, documents are converted into retrievable representations, relevant passages are selected for a user query, and the passages are provided to a generator as context. UniGuide-RAG adopts this broad architecture for institutional-style documents.

### 2.3 Research Gap and Motivation

A campus assistant must address requirements beyond generating a plausible answer. It should preserve the origin of information, expose the document passage supporting a response, avoid inventing details about rules or deadlines, and indicate when the knowledge base does not contain enough evidence. UniGuide-RAG focuses on integrating these requirements into a lightweight document-grounded prototype and measuring them with a manually reviewed test set.

## 3. Preliminaries

### 3.1 Document Collection

Let the document collection be represented by:

\[
D = \{d_1,d_2,\ldots,d_n\}
\]

Each PDF is extracted page by page. The resulting text is cleaned and segmented into overlapping chunks. Each chunk retains metadata containing its source filename, page number and chunk identifier.

### 3.2 Chunking

For a page-level text sequence of length \(T\), chunks are generated using a configurable window size \(s\) and overlap \(o\), where \(0 \leq o < s\). The overlap helps preserve context near chunk boundaries. In the prototype, the default window is 1,100 characters with 180 characters of overlap; these values are configurable and must be included in the final experiment log.

### 3.3 Retrieval

When Gemini embedding access is configured, document chunks and questions are represented as dense vectors. Similarity is computed using cosine similarity:

\[
\operatorname{sim}(q,d_i)=\frac{q\cdot d_i}{\|q\|\|d_i\|}
\]

The top-\(k\) chunks are selected as candidate evidence. When the embedding service is unavailable, the prototype uses TF-IDF vectors and cosine similarity as a lexical baseline. Scores from these two retrieval methods are not directly calibrated against one another.

### 3.4 Grounded Generation

The generator receives the user question and retrieved excerpts. It is instructed to answer only from the supplied excerpts, cite them using source tags and abstain when the excerpts do not support a conclusion. The application also displays the filename and page of each retrieved passage to enable user verification.

## 4. Materials and Methods

### 4.1 System Overview

UniGuide-RAG consists of five stages: document ingestion, text preprocessing, embedding and indexing, evidence retrieval, and grounded answer generation. The Streamlit interface connects these components and provides a document-management view and a conversational question-answering view.

### 4.2 Dataset Preparation

To avoid using private or institution-specific college records, the prototype includes a synthetic corpus created solely for demonstration. The corpus represents a fictional institution named **Northstar Demo University**. It contains four clearly watermarked sample PDFs: a student handbook, an assessment and examination guide, an academic calendar, and a library and IT services guide. All names, offices, contact details, dates, deadlines, opening hours and rules in these documents are invented. They must not be interpreted as real institutional policies.

The synthetic corpus provides a controlled and reproducible environment in which the retrieval pipeline can be tested without exposing student data or implying endorsement by a real institution. The benchmark questions are authored from those same sample PDFs. Each answerable benchmark row records the question, expected source filename, expected page and a short reference answer. The benchmark also includes questions whose answers are intentionally absent from the corpus to examine abstention behaviour. Because the corpus is small and synthetic, performance on it cannot establish generalisation to real institutional documents.

### 4.3 PDF Processing and Chunk Construction

PyMuPDF extracts selectable text from each page. Text is normalised by removing null characters and repeated whitespace while preserving sentence boundaries. Page text is segmented into overlapping chunks. The document name and one-based page number are attached to each chunk, enabling source traceability after retrieval.

Scanned PDFs without a selectable text layer are not processed by the current MVP. Such documents must either be replaced with accessible text PDFs or handled by a separately tested OCR extension.

### 4.4 Embedding and Retrieval Module

The system supports two retrieval configurations. The primary configuration uses Gemini embedding vectors when the API key and embedding model are available. The fallback configuration uses TF-IDF vectors. For a query \(q\), the retriever ranks indexed chunks by cosine similarity and returns the top-\(k\) results. The chosen retrieval backend and configuration must be logged for each experiment.

### 4.5 Evidence-Constrained LLM Generation

The retrieved passages are assembled with source labels such as [S1] and [S2]. The LLM receives the question, source-tagged excerpts and system instructions. It is instructed not to follow instructions that might appear inside source documents, not to use unsupported outside knowledge, and not to invent dates, fees, eligibility requirements or institutional rules. If retrieved evidence is insufficient, the model should state that the information could not be verified from the available documents.

### 4.6 User Interface

The application is implemented using Streamlit. The interface includes a document upload and indexing page, a conversational assistant, a list of indexed sources and an evidence expander. The user can inspect the text passage used for an answer and its original page location.

### 4.7 Experimental Design

The evaluation should compare at least the following configurations:

- TF-IDF retrieval with the same generator.
- Dense Gemini-embedding retrieval with the same generator.
- LLM-only answering without retrieved institution context (controlled baseline, tested carefully).
- Full UniGuide-RAG with source display and abstention policy.

The benchmark should include answerable questions and unanswerable questions. At least two reviewers should independently judge answer correctness and whether the cited passages support the generated claims, where feasible. Any disagreements should be resolved and documented.

### 4.8 Evaluation Metrics

**Hit Rate@k:** proportion of questions for which the expected source appears among the top-\(k\) retrieved chunks.

\[
HitRate@k=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}(r_i\leq k)
\]

**Mean Reciprocal Rank (MRR@k):** average reciprocal rank of the first retrieved chunk from the expected source.

\[
MRR@k=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}
\]

where \(r_i\) is the rank of the first relevant result; it is zero when no relevant result appears in the top-\(k\).

**Answer correctness:** percentage of generated answers judged correct against the reviewed reference answer and source document.

**Citation support rate:** percentage of factual claims in generated answers that are supported by the cited retrieved passages, based on manual review.

**Abstention quality:** evaluate both appropriate abstentions on unanswerable questions and inappropriate abstentions on answerable questions.

**Latency:** measure retrieval time, generation time and total end-to-end response time separately. Report the number of queries and summary statistics (mean and median, and standard deviation where appropriate).

## 5. Experimental Results

**This section must be completed only after running the implementation.** Do not insert estimated or invented measurements.

### 5.1 Experimental Environment

| Item | Value to record |
|---|---|
| Operating system | [Windows/Linux/macOS version] |
| Processor | [CPU model] |
| RAM | [RAM capacity] |
| GPU | [GPU model or none] |
| Python version | [Version] |
| LLM model | [Exact model identifier] |
| Embedding model | [Exact model identifier or TF-IDF] |
| Number of documents | [Measured count] |
| Number of pages | [Measured count] |
| Number of chunks | [Measured count] |
| Benchmark questions | [Measured count] |

### 5.2 Initial Retrieval Results

An initial local evaluation was run using the included synthetic corpus, TF-IDF retrieval and the 12 answerable questions in the benchmark. The expected source document appeared at rank 1 for all 12 questions.

| Retrieval configuration | Hit Rate@5 | MRR@5 | Query count |
|---|---:|---:|---:|
| TF-IDF (synthetic demo benchmark) | 1.000 | 1.000 | 12 |
| Gemini embeddings | Not measured | Not measured | - |

These values are preliminary software-test results, not evidence of performance on real institutional material. The benchmark is small, synthetic and authored from the same four documents used by the prototype. An independent, larger test set is required before drawing broader conclusions. The two rows marked `NO_ANSWER` are excluded from source-ranking metrics and are evaluated separately for abstention. On the current TF-IDF run, their top similarity scores were approximately 0.150 and 0.100; the demo fallback threshold is set to 0.17 so both queries should trigger abstention. Similarity scores are not probabilities and this threshold must be retuned for a different corpus or retrieval backend.

### 5.3 Answer Quality and Evidence Attribution

LLM-generated answer correctness and citation support have not yet been measured with a live Gemini API session. They must be evaluated by comparing each answer with the reference text and checking whether every factual claim is supported by its cited passage. The retrieval-only abstention check on the two `NO_ANSWER` questions returned top TF-IDF scores below the configured 0.17 threshold (2/2 expected abstentions in this small test). This is a limited prototype check, not a calibrated confidence estimate.

| Configuration | Answer correctness | Citation support rate | Abstention quality |
|---|---:|---:|---:|
| LLM-only baseline | Not measured | N/A | Not measured |
| RAG without abstention instruction | Not measured | Not measured | Not measured |
| Full UniGuide-RAG with Gemini | Not measured | Not measured | To be tested end-to-end |

### 5.4 Response Time

End-to-end latency has not yet been benchmarked. Report measured retrieval time, generation time and total latency. Include query count, mean, median and standard deviation. Distinguish API/network latency from local processing time.

### 5.5 Ablation Study

Evaluate the contribution of individual components by disabling one component at a time:

1. Remove dense embeddings and use TF-IDF.
2. Remove source-tagged passages from the prompt.
3. Remove the abstention instruction.
4. Remove page-aware metadata from the interface.

The effect of each change should be assessed using the same benchmark questions and evaluation procedure.

## 6. Discussion

Discuss whether retrieval returns the correct sample document, whether generated statements are supported by citations, and whether abstention reduces unsupported answers without rejecting too many answerable questions. Explain how document quality, missing publication dates, conflicting notices and stale content affect system reliability. Discuss the trade-off between dense embedding retrieval and the simpler TF-IDF fallback.

## 6.1 Limitations

The current prototype processes text-searchable PDFs and does not perform OCR on scanned documents. It relies on the quality and completeness of the documents supplied by the operator. Retrieval similarity is not a calibrated probability of answer correctness. The LLM can still misunderstand evidence or produce unsupported statements, so important policy decisions must be checked against the original document and the relevant information owner. The prototype is not an official institution service and does not authenticate document validity.

## 7. Conclusion and Future Work

This paper presents UniGuide-RAG, a prototype campus information assistant that combines page-aware PDF ingestion, retrieval, evidence-constrained LLM generation and source inspection. The system is designed to answer questions using sample documents and to abstain when the available evidence is insufficient. The research evaluation will quantify retrieval effectiveness, answer correctness, citation support, abstention quality and response latency after the benchmark is populated and experiments are completed.

Future work may include OCR for scanned documents, explicit document-version and effective-date ranking, multilingual evaluation on institutionally reviewed question sets, role-based access control, a larger benchmark, human feedback and deployment within an authenticated institution environment.

## References

[1] P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” *Advances in Neural Information Processing Systems*, vol. 33, 2020. arXiv:2005.11401. https://doi.org/10.48550/arXiv.2005.11401.

[2] Google, “Google Gen AI Python SDK documentation.” https://googleapis.github.io/python-genai/ (accessed 25 November 2026).

[3] Streamlit, “Session State documentation.” https://docs.streamlit.io/develop/concepts/architecture/session-state (accessed 25 November 2026).

---

## Appendix A. Reproducibility Checklist

- [ ] Record all source documents and their version/effective dates.
- [ ] Record exact embedding and LLM model identifiers.
- [ ] Record chunk size, overlap and top-k.
- [ ] Record retrieval threshold and selected retrieval backend.
- [ ] Build and freeze a manually reviewed question benchmark.
- [ ] Use the same benchmark for every baseline and ablation.
- [ ] Save raw query-level outputs and latency measurements.
- [ ] Report failed queries and unsupported-answer cases.
- [ ] Include application screenshots and the architecture diagram.
