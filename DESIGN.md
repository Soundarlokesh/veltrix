# Design

## 1. Ingestion and OCR

`app/ingestion/parsers.py` reads every manifest entry. It handles PDF (PyMuPDF), DOCX (python-docx, walking body elements in order), Markdown and HTML (BeautifulSoup, footers and scripts removed). Each format becomes ordered items (heading, text, table, page), then **segments** that keep their heading and page. Headings are found from structure where it exists (DOCX heading styles, HTML `h1`–`h6`, Markdown `#`). PDFs have no structure, so numbered lines such as `6. Carrying annual leave forward` are headings. Tables are found with PyMuPDF's table detector and emitted as rows, so cell values stay together.

A PDF page with almost no text and an image is **scanned**. Detection runs without OCR, so the scanned memo is always identified. Such a page is rendered at 300 dpi and read with Tesseract. If Tesseract is missing, or OCR returns nothing, ingestion **fails** with an actionable message. The store is never written, and `/health` stays 503. Nothing is silently skipped.

Holidays come from the calendar tables. A row counts only if it has a full date and a weekday. The category comes from the section heading and the row text: `restricted`, `company` or `public`. Dates are never typed into code.

## 2. Chunking and retrieval

Chunks never span two headings. A chunk is at most 200 `cl100k_base` tokens, below the 256-token limit of the embedding model. Each chunk stores doc_id, title, doc_type, regions, effective date, access level, heading, page range, OCR flag and locator (heading, else `p. N`, else title). Embeddings use `all-MiniLM-L6-v2` on CPU, stored in a FAISS flat inner-product index. Vectors are normalised, so the score is cosine similarity. The index is rebuilt when a fingerprint of chunk text, metadata, embedder and parser version changes.

Retrieval is filtered **before** ranking. The search predicate (`chunk_allowed`) removes chunks the requester cannot read, that do not apply to their region, or that are not yet in effect. Upcoming documents are searched separately with their own predicate.

## 3. Precedence

`app/policy/precedence.py` applies TASK §5.1 in order: authority, then regional scope, then recency. A test is applied only if values still differ. All three tests run so that an overridden document that states the same value is not cited. If values still differ, the result is a **conflict**: all values and documents are returned, and nothing is chosen.

The model proposes rules (topic, value, chunk id). Code decides the outcome. Values are compared case-insensitively. A rule is dropped unless its chunk was in the context. Upcoming changes are mentioned only if the document is readable, applies to the region, and would win on its effective date.

**Limitation:** grouping rules by topic depends on the model's topic wording. Two documents that word the same rule differently would not be compared. This is the largest accuracy risk and needs live-model evaluation.

## 4. Access control

Access is enforced in code, never in prompts.

- `app/db/employees.py` reads role and region from `veltrix.db` (read-only). Message text is never used for identity.
- `app/policy/access.py` maps roles to readable access levels (TASK §3.1). An unknown role reads nothing.
- `app/policy/filtering.py` creates `EligibleChunk` and `UpcomingChunk` objects, which cannot be built by hand.
- `app/llm/context.py` `build_context()` accepts only those types, so a raw chunk cannot reach the model.
- `app/tools/permissions.py` decides who may see whose leave. Employees see their own data. Managers see direct reports only. HR sees one employee at a time. Only managers call `get_team_leave`. Unknown and forbidden targets get the same refusal text.
- Each tool in `app/tools/leave_tools.py` checks permission itself. Apply and cancel accept no target employee.
- Chunks that contain instructions aimed at an AI assistant are quarantined (`app/security/injection.py`) and never enter a context.
- The router refuses manipulation and bulk requests before any model or tool runs.

## 5. Leave writes

`app/leave/service.py` validates every rule inside a `BEGIN IMMEDIATE` transaction. Apply inserts the request and deducts the balance together, and a conditional `UPDATE` must change exactly one row. Cancel sets the status and restores the balance together. Any failure rolls back, and the user sees "No changes were made".

Confirmation is a two-step flow. A proposal is stored in `state.db` (bound to session and employee, expires after 10 minutes). A confirmation first claims the row atomically (`pending` to `executing`), so a repeated or concurrent confirmation cannot run twice. The tool then re-validates against the current database state.

## 6. Personal data

Names, emails and phone numbers are not returned by any tool. Emails are masked as `r***@domain`, and phone-like numbers are removed from answers. The model receives the redacted question and only the policy context. Logs pass through a sanitiser filter, and they use employee IDs and hashed session tags.

## 7. Failure modes found and mitigated

| Failure | Mitigation |
|---|---|
| faiss and torch load two OpenMP runtimes on Windows, aborting the process | `KMP_DUPLICATE_LIB_OK` on Windows, with a test that FAISS matches brute-force search |
| `executescript` committed mid-write | One statement per `execute` in an explicit transaction |
| Overridden document cited as a source | All precedence tests run before citing |
| Date-only and weekday phrases misread | Explicit rules; ambiguous phrases trigger a question |
| A changed balance or overlap between proposal and confirmation | Re-validation at confirmation, with a test for each case |
| Double confirmation | Atomic claim of the pending row |
| BOM in the manifest made startup fail | Loader reads `utf-8-sig`, with a test |
| Model cites a chunk it was not given | Citation rejected; answer becomes no-answer |
| Model reply is not JSON | No answer, with a safe message |
| Model timeout or error | Safe no-answer, with no error detail sent to the user |

## 8. At 10,000 documents

Expected first failures:

1. **Flat FAISS search** scores every vector per query, so latency grows linearly. Fix: an approximate index (HNSW or IVF) with metadata pre-filters stored in the index, or a vector database such as pgvector or Qdrant with payload filters.
2. **Ingestion time** parses and embeds everything at startup, so readiness could take hours. Fix: incremental ingestion keyed by file hash, and a persisted index.
3. **Topic grouping for precedence** becomes less reliable as near-duplicate documents multiply. Fix: a curated topic index built at ingestion, reviewed by HR, and tested against known conflicts.
4. **SQLite write locking** serialises writes. Fix: PostgreSQL with row-level locks, while keeping the same transaction boundaries.
5. **Context budget**: more documents per question make the 3,000-token limit tighter. Fix: rerank before packing, and keep the same deterministic filters.
