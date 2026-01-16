# System Architecture (local)

            [Company Policy Documents (.txt)]
                          │
                          ▼
                  Document Loader
           (Chunking paragraphs / sections,
             overlap for context)
                          │
                          ▼
                Embedding Generator
       (SentenceTransformers → 384-d embeddings)
                          │
                          ▼
                 FAISS Vector Store
        (IndexFlatIP, normalized embeddings)
                          │
                          ▼
             Query / Retrieval Module
       (Retrieve top-k relevant chunks,
        optional filtering by keywords)
                          │
                          ▼
            Rule-Based Answer Synthesizer
           (Bullet-style answers, grounded)
                          │
                          ▼
                    Output Answer
         
# System Architecture (Gemini)

            [Company Policy Documents (.txt)]
                          │
                          ▼
                  Document Loader
           (Chunking paragraphs / sections,
             overlap for context)
                          │
                          ▼
                Embedding Generator
       (SentenceTransformers → 384-d embeddings)
                          │
                          ▼
                 FAISS Vector Store
        (IndexFlatIP, normalized embeddings)
                          │
                          ▼
             Query / Retrieval Module
       (Retrieve top-k relevant chunks,
        optional filtering by keywords)
                          │
                          ▼
                  Gemini Cloud
           (gemini-flash-latest via API)
           - Generates grounded answer
           - Requires API key
                          │
                          ▼
                     Output Answer
         (Grounded, bullet-style via Gemini)

## Notes:
* All local components: Document Loader, Chunking, Embeddings, FAISS, Retrieval.
* Answer generation: Fully dependent on Gemini in the cloud.

# Design Choices

## Chunking Strategy:
Large chunks (500–1000 characters) to preserve eligibility rules and exceptions.
Overlap (100 characters) maintains continuity across policy sections.

## Embedding Model:
all-MiniLM-L6-v2 chosen for speed, local execution, and semantic relevance.
Produces normalized 384-dimensional embeddings for FAISS cosine similarity search.

## Vector Search Threshold:
Only include chunks with cosine similarity ≥ 0.5.
Top-k retrieval (k=3–5) balances completeness and precision.

## Answer Generation:
Keyword-based extraction from retrieved chunks ensures answers are grounded in documents.
Deduplication and bullet-limiting improve readability.

## Local First Approach:
Entire retrieval, embeddings, and answer synthesis run without external calls, ensuring privacy and security.

# Usage

## Run using Gemini
```bash
python rag_system.py
```
## Run locally without Gemini
```bash
python rag_local_only.py
```
## Ask a single question
```bash
python local_rag.py --question "How is bonus eligibility determined?"
```
## Run sample evaluation
```bash
python local_rag.py --eval
```
# Screenshots
<img width="1223" height="822" alt="Screenshot 2026-01-16 170041" src="https://github.com/user-attachments/assets/7c6be17e-9c21-4caa-a99a-3b6cb36be6d2" />

<img width="1226" height="448" alt="Screenshot 2026-01-16 170142" src="https://github.com/user-attachments/assets/7bf91721-e74e-4d3e-bad1-da0cfa276320" />

<img width="1224" height="658" alt="Screenshot 2026-01-16 170204" src="https://github.com/user-attachments/assets/64f8b7e3-3292-4f08-9bb4-014c1f87a3ca" />





