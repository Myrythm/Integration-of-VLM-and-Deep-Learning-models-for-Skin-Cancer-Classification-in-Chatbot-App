# Graph Report - .  (2026-07-26)

## Corpus Check
- 148 files · ~63,209 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 596 nodes · 1198 edges · 47 communities (41 shown, 6 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 109 edges (avg confidence: 0.77)
- Token cost: 300,619 input · 0 output

## Community Hubs (Navigation)
- App Bootstrap and Routing
- Chat SSE and Citations
- Config, Classifier and LLM Backends
- RAG Chain and Session Memory
- Retriever Adapters and Evidence Filter
- Detection Schemas and Upload API Tests
- Chat Route Integration Tests
- Codebase Review Findings
- UI Design System and Templates
- Vector Store Providers
- GPT-4o Vision Image Validation
- Structured Query Logging
- Embedding Cache
- PubMed Fetch and Parse
- RAG Design Spec and Research Basis
- Safety Gate and Disclaimer Policy
- Clinical Knowledge Base Sources
- Ingestion and KB Expansion
- OpenAI Embedder
- Layering Rule and Label-Order Fix
- Chat Frontend Script
- Lesion Types and Model Labels
- Eval Metrics and Threshold Tuning
- SSE Streaming and Citation Markers
- Fail-Closed Evidence Policy
- Swappable Backend Architecture
- Prompt Injection and Disclaimer UI
- Bilingual Knowledge Base Strategy
- Async Streaming Flow Overview
- Pinned ML Dependency Stack
- Module Boundaries and FastAPI Rewrite
- Upload Frontend Script
- Early Detection Guidance
- Idempotent Ingest Safety
- Model Weights and Env Config
- Knowledge Base Init Script
- Eval Runner Script

## God Nodes (most connected - your core abstractions)
1. `Settings` - 75 edges
2. `Codebase Review (Efficiency, API Contract, Maintainability, Scalability)` - 28 edges
3. `get_settings()` - 23 edges
4. `EvidenceFilteredRetriever` - 20 edges
5. `SessionMemory` - 19 edges
6. `initialize_app_state()` - 18 edges
7. `ChromaLangChainRetriever` - 18 edges
8. `get_vector_store()` - 18 edges
9. `DetectionResult` - 17 edges
10. `classify_skin_image()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Evidence Filter Fails Closed Below Similarity Threshold` --semantically_similar_to--> `Fail-Closed on VLM Failure (HTTP 503)`  [INFERRED] [semantically similar]
  CLAUDE.md → docs/superpowers/plans/2026-07-05-vlm-image-validation-and-label-fix.md
- `Class-Strategy Dark Mode with localStorage Persistence` --conceptually_related_to--> `SkinVision — Skin-Cancer RAG Chatbot`  [AMBIGUOUS]
  docs/superpowers/plans/2026-06-20-ui-and-folder-restructure.md → README.md
- `SKIN_CANCER_LABELS 4-Class Output Order` --shares_data_with--> `EfficientNetB3 4-Class Lesion Classification`  [INFERRED]
  model/README.md → README.md
- `SHA256 Query Hashing for Logs` --semantically_similar_to--> `F4: SSE Error Path Leaks Internals, Never Logs, Never Emits done`  [INFERRED] [semantically similar]
  docs/superpowers/specs/2026-06-20-skin-cancer-rag-design.md → review.md
- `PubMed Knowledge Base Expansion Design` --semantically_similar_to--> `F19: PubMed Parsing Truncates Abstracts and Mangles Dates`  [INFERRED] [semantically similar]
  docs/superpowers/specs/2026-07-04-pubmed-kb-expansion-design.md → review.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Image Upload Validation and Classification Pipeline** — claude_upload_flow, docs_superpowers_plans_2026_07_05_vlm_image_validation_and_label_fix_vlm_validation_gate, docs_superpowers_plans_2026_07_05_vlm_image_validation_and_label_fix_fail_closed_policy, docs_superpowers_plans_2026_07_05_vlm_image_validation_and_label_fix_decode_precheck, readme_efficientnetb3_classification, model_readme_skin_cancer_labels [EXTRACTED 1.00]
- **Medical Safety and Compliance Stack** — claude_safety_gate, claude_mandatory_disclaimer, docs_architecture_three_layer_disclaimer, docs_superpowers_plans_2026_06_20_skin_cancer_rag_prompt_injection_resistance, readme_educational_not_diagnostic [INFERRED 0.85]
- **Grounded Retrieval and Citation Flow** — readme_lcel_rag_pipeline, claude_evidence_filter_fail_closed, readme_verifiable_citations, docs_knowledge_base_idempotent_ingestion, claude_swappable_backends [INFERRED 0.85]
- **Three-Layer Mandatory Disclaimer Enforcement** — docs_superpowers_specs_2026_06_20_skin_cancer_rag_design_three_layer_disclaimer, docs_superpowers_specs_2026_06_20_skin_cancer_rag_design_prompt_injection_defense, templates_chat_disclaimer_banner, templates_partials_disclaimer_bar_component, review_f7_blocked_path_skips_compliance, tests_eval_report_metrics_table [INFERRED 0.85]
- **Image Upload Validation and Classification Flow** — docs_superpowers_specs_2026_07_05_vlm_image_validation_and_label_fix_design_local_decode_precheck, docs_superpowers_specs_2026_07_05_vlm_image_validation_and_label_fix_design_vlm_validation_gate, docs_superpowers_specs_2026_07_05_vlm_image_validation_and_label_fix_design_validation_unavailable_error, docs_superpowers_specs_2026_07_05_vlm_image_validation_and_label_fix_design_label_order_fix, docs_superpowers_specs_2026_07_05_vlm_image_validation_and_label_fix_design_validation_status_field, templates_upload_error_card, templates_upload_result_card, review_f11_upload_unbounded_buffer, review_f17_classifier_output_unguarded [EXTRACTED 1.00]
- **SSE /api/chat Public Contract Findings** — review_f4_sse_error_path, review_f8_untyped_sse_schema, review_f9_citations_not_filtered_to_cited, review_f14_retrieve_step_duck_typing, review_f15_no_llm_stream_timeout, review_f16_sse_buffering_and_health, review_f22_quadratic_token_accumulation, docs_superpowers_specs_2026_06_20_ui_and_folder_restructure_design_fetch_readablestream_sse [INFERRED 0.85]

## Communities (47 total, 6 thin omitted)

### Community 0 - "App Bootstrap and Routing"
Cohesion: 0.07
Nodes (45): get_settings(), FastAPI, HTMLResponse, JSONResponse, health(), lifespan(), Liveness probe: the process is up. Does not assert app state is ready., Readiness probe: app state (RAG chain) is initialized. (+37 more)

### Community 1 - "Chat SSE and Citations"
Cohesion: 0.06
Nodes (49): _answer_stream(), _blocked_stream(), chat(), Render the detection into the system prompt safely.      The detection comes f, _render_detection(), _sse(), ChatChunk, ChatRequest (+41 more)

### Community 2 - "Config, Classifier and LLM Backends"
Cohesion: 0.09
Nodes (38): BaseSettings, Settings, ndarray, classify_skin_image(), _get_model(), preprocess_image(), _softmax(), get_llm_provider() (+30 more)

### Community 3 - "RAG Chain and Session Memory"
Cohesion: 0.07
Nodes (32): BaseMessage, build_rag_chain(), format_docs(), BaseChatModel, BaseRetriever, ChatPromptTemplate, Document, Runnable (+24 more)

### Community 4 - "Retriever Adapters and Evidence Filter"
Cohesion: 0.11
Nodes (25): AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun, ChromaLangChainRetriever, EvidenceFilteredRetriever, Any, BaseRetriever, Document, Adapter: bridges Embedder + VectorStoreProvider to LangChain retriever interface (+17 more)

### Community 5 - "Detection Schemas and Upload API Tests"
Cohesion: 0.11
Nodes (22): DetectionResult, BaseModel, # NOTE: these must stay in sync with services/image/classifier.SKIN_CANCER_LABEL, ImageUploadResponse, BaseModel, _make_test_image_bytes(), mocked_classifier(), mocked_validator() (+14 more)

### Community 6 - "Chat Route Integration Tests"
Cohesion: 0.26
Nodes (21): TestClient, _build_chain_mock(), _make_patches(), _parse_sse(), _post(), Document, _retriever_end_event(), test_blocked_query_appends_disclaimer_and_logs() (+13 more)

### Community 7 - "Codebase Review Findings"
Cohesion: 0.16
Nodes (19): Bounded 6-Turn Session Memory, SHA256 Query Hashing for Logs, Cheap Local Decode Pre-Check, Codebase Review (Efficiency, API Contract, Maintainability, Scalability), F11: Upload Buffers Unlimited Bytes and Leaks Paths, F12: session_id Is an Unvalidated Free-Form String, F13: get_settings() Re-Parses .env on Every Call, F15: No Timeout Around the LLM Stream (+11 more)

### Community 8 - "UI Design System and Templates"
Cohesion: 0.16
Nodes (18): Anti-AI-Slop Design Principia, Clinical Light/Dark Color Palette, Tailwind darkMode:'class' Toggle Strategy, Indonesian-Only UI Chrome, UI + Folder Restructure Design Spec, Tailwind via CDN (No Build Step), Three-Page MVP UI (Home, Upload, Chat), Bilingual Image Rejection Message (+10 more)

### Community 9 - "Vector Store Providers"
Cohesion: 0.15
Nodes (8): ChromaProvider, PineconeProvider, Drop the collection and recreate it empty so the provider stays usable., Stub: Phase 3 placeholder. Implement with `langchain-pinecone` when needed., Path, settings(), test_delete_collection_wipes_data_and_allows_reuse(), test_upsert_and_similarity_search_roundtrip()

### Community 10 - "GPT-4o Vision Image Validation"
Cohesion: 0.28
Nodes (15): Exception, Raised when the vision validation cannot be completed (API/network/timeout/confi, Ask GPT-4o Vision whether the image is a skin lesion.      Returns one of "valid, validate_skin_image(), ValidationUnavailableError, _mock_client_returning(), Build an AsyncOpenAI-shaped mock whose chat completion returns `text`., _settings() (+7 more)

### Community 11 - "Structured Query Logging"
Cohesion: 0.23
Nodes (13): hash_query(), log_query(), SHA256 hash of query for PII-safe logging., # NOTE: TimedRotatingFileHandler is not multi-process safe — under, Emit a structured log entry for a single query., setup_logging(), Repeated calls must not stack handlers (e.g. on worker reload or test rerun)., test_hash_query_deterministic() (+5 more)

### Community 12 - "Embedding Cache"
Cohesion: 0.26
Nodes (7): EmbeddingCache, Path, File-based cache for embeddings, keyed by SHA256 of input text., Path, test_cache_miss_returns_none(), test_cache_persists_across_instances(), test_cache_roundtrip()

### Community 13 - "PubMed Fetch and Parse"
Cohesion: 0.35
Nodes (8): main(), Fetch PubMed abstracts and save as JSON for ingestion. Run: python data/knowled, _efetch(), _esearch(), fetch_pubmed_abstracts(), parse_pubmed_xml(), test_fetch_pubmed_abstracts_uses_api(), test_parse_pubmed_xml()

### Community 14 - "RAG Design Spec and Research Basis"
Cohesion: 0.18
Nodes (11): BriefContext (2412.15271), Pre-LLM Dangerous Query Classifier, i-MedRAG (2408.00727), LCEL RAG Chain (retriever | prompt | llm), LLMProvider Protocol + Factory, Single Multilingual Prompt Template, Skin Cancer RAG-Enhanced Chatbot Design Spec, F10: Token Usage Logged as 0/0 (+3 more)

### Community 15 - "Safety Gate and Disclaimer Policy"
Cohesion: 0.20
Nodes (10): Mandatory Idempotent Medical Disclaimer, Safety Gate: Canned Bilingual Refusals, Three-Layer Disclaimer, Prompt-Injection-Resistant System Rules, Heuristic Query Danger Classifier, Three-Layer Mandatory Disclaimer Design, Cheap Local Pillow Decode Pre-Check Before VLM Call, Unrecognized VLM Reply Degrades to 'uncertain' (+2 more)

### Community 16 - "Clinical Knowledge Base Sources"
Cohesion: 0.24
Nodes (10): ABCDE Warning Signs (AAD), Melanoma (AAD Overview), ABCDE Rule for Melanoma Detection (DermNet), Melanoma (MedlinePlus), Skin Cancer Treatment Options by Type, AAD Patient-Education Source, DermNet NZ Source (CC BY-NC-ND), MedlinePlus Public-Domain Source (+2 more)

### Community 17 - "Ingestion and KB Expansion"
Cohesion: 0.22
Nodes (10): Cross-Lingual Embedder (text-embedding-3-small), Two-Tier Knowledge Base (Guidelines + PubMed), MIRAGE / MedRAG Benchmark (2402.13178), Semantic Chunking Ingestion Pipeline, VectorStoreProvider Protocol + Factory, Empty Knowledge Base Root Cause, SHA256 Idempotent Chunk IDs, Incremental Ingest Without --rebuild (+2 more)

### Community 18 - "OpenAI Embedder"
Cohesion: 0.33
Nodes (5): OpenAIEmbedder, settings(), test_aembed_query_uses_async_client(), test_embed_documents_batches(), test_embed_query_returns_vector()

### Community 19 - "Layering Rule and Label-Order Fix"
Cohesion: 0.22
Nodes (9): Framework-Agnostic RAG Module Boundary, utils/ to services/ Refactor, SKIN_CANCER_LABELS Order Fix, VLM Image Validation + Label-Order Fix Design, GPT-4o Vision Lesion Validation Gate, F17: Classifier Output Unguarded Against Shape and Logit Surprises, F21: CLAUDE.md, architecture.md, README Describe a Codebase That No Longer Exists, F5: Client detection Field Is a Prompt-Injection Channel (+1 more)

### Community 20 - "Chat Frontend Script"
Cohesion: 0.47
Nodes (8): appendUserMessage(), createBotBubble(), displayInitialResult(), handleSend(), processText(), removeTypingIndicator(), scrollToBottom(), showTypingIndicator()

### Community 21 - "Lesion Types and Model Labels"
Cohesion: 0.25
Nodes (8): POST /api/upload Request Flow, Actinic Keratosis (Pre-Cancer), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Classifier Label Order Is Ground Truth from Trained Model, EfficientNetB3 (frozen) + Dense(256) + Dropout + Dense(4, softmax), Preprocessing Pipeline (PIL to RGB to 224x224 to expand_dims), SKIN_CANCER_LABELS 4-Class Output Order

### Community 22 - "Eval Metrics and Threshold Tuning"
Cohesion: 0.36
Nodes (8): 30-Question Bilingual Eval Gold Set, Evidence Filter (Cosine Similarity Threshold), RAG_SIMILARITY_THRESHOLD Config Discrepancy, ValidationUnavailableError (Fail-Closed Signal), F6: Evidence Filter Fails Open When a Chunk Has No Score, Eval Report, Eval Metrics Targets Table, Similarity Threshold Sweep

### Community 23 - "SSE Streaming and Citation Markers"
Cohesion: 0.25
Nodes (8): Inline [n] Citation Markers, MMed-RAG (2410.13085), SSE Token Streaming via StreamingResponse, fetch() + ReadableStream SSE Parsing, F16: Missing SSE Anti-Buffering Headers; /health Lies Before Init, F8: SSE Chunk Schema Untyped; CitationOut Dead, F9: Citation Events Report Every Retrieved Chunk, Chat SSE Mount Points (#chat-area, #chat-input, #send-btn)

### Community 24 - "Fail-Closed Evidence Policy"
Cohesion: 0.29
Nodes (7): Evidence Filter Fails Closed Below Similarity Threshold, Evidence-Filtered Retriever (Cosine Threshold), Similarity Threshold Is Report-Only, Not Auto-Tuned, Fail-Closed on VLM Failure (HTTP 503), LCEL RAG Pipeline (retrieve | prompt | llm), Verifiable Citations Filtered to Cited [n] Markers, LangChain 0.3 Stack (langchain, langchain-openai, langchain-community)

### Community 25 - "Swappable Backend Architecture"
Cohesion: 0.29
Nodes (7): Protocol + Factory Swappable Backends, Module Map of FastAPI Web App, Anti-AI-Slop Clinical Design Constraint, Class-Strategy Dark Mode with localStorage Persistence, utils/ to services/ Rename Refactor, Roadmap: Ollama / vLLM / Pinecone Stub Backends, SkinVision — Skin-Cancer RAG Chatbot

### Community 26 - "Prompt Injection and Disclaimer UI"
Cohesion: 0.38
Nodes (7): Prompt Injection Defense, Rethinking RAG for Medicine (2511.06738), Three-Layer Mandatory Medical Disclaimer, Chat Persistent Disclaimer Banner, marked + DOMPurify Response Rendering, chat.html Consultation Page, disclaimer_bar.html Partial

### Community 27 - "Bilingual Knowledge Base Strategy"
Cohesion: 0.40
Nodes (6): Bilingual (id/en) Language Detection with One Prompt Template, PubMed Abstract Source (Filtered Query), Knowledge-Base Source Licensing Table, English-Only Knowledge Base with LLM-Level Multilingual Generation, DEFAULT_QUERY Relaxation (Drop patient education MeSH Clause), PubMed Knowledge Base Expansion Plan

### Community 28 - "Async Streaming Flow Overview"
Cohesion: 0.33
Nodes (6): POST /api/chat SSE Event-Translation Flow, Skin-Cancer RAG Chatbot Overview, Token Accounting via ChatOpenAI stream_usage, SSE over POST via fetch + ReadableStream (not EventSource), VLM Image Validation + Classifier Label-Order Fix Plan, Fully Async SSE Streaming with asyncio.to_thread Offload

### Community 29 - "Pinned ML Dependency Stack"
Cohesion: 0.33
Nodes (6): TensorFlow/Python ABI Compatibility Constraint, EfficientNetB3 4-Class Lesion Classification, Privacy: Local Chroma + Hashed Query Logging, chromadb==0.5.23 Pin, Pinned Runtime Dependency Set, tensorflow==2.15.0 Pin

### Community 30 - "Module Boundaries and FastAPI Rewrite"
Cohesion: 0.40
Nodes (5): Lifespan-Managed Singleton Wiring, Layering Rule: services/rag is Framework-Agnostic, Tests Bypass Wiring by Patching app_state Module Globals, Import Boundaries Between routes, schemas, services, Full Flask to FastAPI Rewrite

## Ambiguous Edges - Review These
- `Tests Bypass Wiring by Patching app_state Module Globals` → `Full Flask to FastAPI Rewrite`  [AMBIGUOUS]
  CLAUDE.md · relation: conceptually_related_to
- `SkinVision — Skin-Cancer RAG Chatbot` → `Class-Strategy Dark Mode with localStorage Persistence`  [AMBIGUOUS]
  docs/superpowers/plans/2026-06-20-ui-and-folder-restructure.md · relation: conceptually_related_to

## Knowledge Gaps
- **24 isolated node(s):** `init_kb.sh script`, `run_eval.sh script`, `Bilingual (id/en) Language Detection with One Prompt Template`, `Token Accounting via ChatOpenAI stream_usage`, `Verifiable Citations Filtered to Cited [n] Markers` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Tests Bypass Wiring by Patching app_state Module Globals` and `Full Flask to FastAPI Rewrite`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `SkinVision — Skin-Cancer RAG Chatbot` and `Class-Strategy Dark Mode with localStorage Persistence`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Settings` connect `Config, Classifier and LLM Backends` to `App Bootstrap and Routing`, `Retriever Adapters and Evidence Filter`, `Chat Route Integration Tests`, `Vector Store Providers`, `GPT-4o Vision Image Validation`, `OpenAI Embedder`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `get_settings()` connect `App Bootstrap and Routing` to `Chat SSE and Citations`, `Config, Classifier and LLM Backends`, `GPT-4o Vision Image Validation`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `SessionMemory` connect `RAG Chain and Session Memory` to `App Bootstrap and Routing`, `Chat SSE and Citations`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `Settings` (e.g. with `ValidationUnavailableError` and `Embedder`) actually correct?**
  _`Settings` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `EvidenceFilteredRetriever` (e.g. with `_AsyncOnlyBase` and `_AsyncOnlyEmbedder`) actually correct?**
  _`EvidenceFilteredRetriever` has 6 INFERRED edges - model-reasoned connections that need verification._