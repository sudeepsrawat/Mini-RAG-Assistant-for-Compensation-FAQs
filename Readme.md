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
python rag_system.py

## Run locally without Gemini
python rag_local_only.py


