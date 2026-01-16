"""
FULLY LOCAL RAG System for Compensation FAQs
- No APIs required
- Uses Sentence Transformers for embeddings
- Uses FAISS for similarity search
- Rule-based synthesis for grounded answers
"""

import os
from typing import List, Dict, Tuple
import re

from document_loader import DocumentLoader
from embeddings import EmbeddingGenerator, VectorStore


class LocalRAG:

    def __init__(self, documents_dir: str = "data"):
        self.documents_dir = documents_dir
        self.loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_generator)

        self.load_and_index_documents()
        print("\n" + "=" * 60)
        print("✅ LOCAL RAG SYSTEM READY")
        print("=" * 60)

    def load_and_index_documents(self):
        """Load and chunk documents, build FAISS index"""
        print(f"📂 Loading documents from {self.documents_dir}...")
        docs = self.loader.load_documents(self.documents_dir)
        print(f"   ✅ Loaded {len(docs)} documents")

        print("   🔄 Chunking documents...")
        chunks = self.loader.chunk_documents(docs)
        print(f"   ✅ Created {len(chunks)} chunks")

        print("   📈 Building vector store...")
        self.vector_store.add_documents(chunks)
        print("   ✅ Vector store ready")

    def retrieve_context(self, query: str, k: int = 3) -> List[Tuple[dict, float]]:
        """Retrieve top-k relevant chunks from vector store"""
        results = self.vector_store.search(query, k=k)
        print(f"\n🔍 Retrieved {len(results)} relevant chunks for query: '{query}'")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"  Chunk {i} (Similarity: {score:.2f}) | Source: {chunk['source']}")
            preview = chunk['text'][:200] + ("..." if len(chunk['text']) > 200 else "")
            print(f"    Text Preview: {preview}")
        return results

    def generate_answer(self, query: str, context_chunks: List[Tuple[dict, float]]) -> str:
        """Generate a grounded, bullet-style answer using only local documents"""
        if not context_chunks:
            return "The information is not available in the provided documents."

        # Define keywords relevant to HR queries
        keywords = ['bonus', 'promotion', 'salary', 'role change', 'variable pay',
                    'eligible', 'eligibility', 'criteria', 'policy', 'guideline', 'process', 'factors']

        # Extract relevant lines from top chunks
        answer_lines = []
        sources = set()
        for chunk, _ in sorted(context_chunks, key=lambda x: x[1], reverse=True):
            sources.add(chunk['source'])
            for line in chunk['text'].split('\n'):
                line_clean = line.strip()
                line_lower = line_clean.lower()
                if any(k in line_lower for k in keywords) and len(line_clean) > 15:
                    answer_lines.append(line_clean)

        if not answer_lines:
            return "The information is not available in the provided documents."

        # Deduplicate while preserving order
        seen = set()
        bullets = []
        for line in answer_lines:
            line_clean = re.sub(r'\s+', ' ', line).strip()
            line_clean = line_clean.lstrip(':•-–— ')  # remove leading punctuation/bullets
            if line_clean not in seen:
                bullets.append(f"* {line_clean}")
                seen.add(line_clean)

        # Limit to top 8 bullets for readability
        bullets = bullets[:8]

        # Compose final answer
        answer = "Based on the company documents:\n\n" + "\n".join(bullets)
        answer += f"\n\nSources: {', '.join(sorted(sources))}"

        return answer

    def ask(self, question: str) -> Dict[str, any]:
        """Main method to ask a question"""
        print("\n" + "=" * 60)
        print(f"❓ QUESTION: {question}")
        print("=" * 60)

        context_chunks = self.retrieve_context(question, k=3)
        answer = self.generate_answer(question, context_chunks)

        print("\n=== ANSWER ===")
        print(answer)

        return {
            "question": question,
            "answer": answer,
            "sources": list(sorted({chunk["source"] for chunk, _ in context_chunks})),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local RAG System")
    parser.add_argument("--eval", action="store_true", help="Run sample evaluation questions")
    parser.add_argument("--question", type=str, help="Ask a specific question")
    args = parser.parse_args()

    rag = LocalRAG()

    sample_questions = [
        "How is bonus eligibility determined?",
        "What happens during a mid year role change?",
        "Are promotions tied to performance ratings?",
        "How often are salaries revised?",
        "What factors affect variable pay?"
    ]

    if args.eval:
        for i, q in enumerate(sample_questions, 1):
            print(f"\n--- Question {i} ---")
            rag.ask(q)
    elif args.question:
        rag.ask(args.question)
    else:
        print("\nInteractive mode. Type your question or 'exit' to quit.\n")
        while True:
            q = input("> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                rag.ask(q)
            else:
                print("Please enter a question.")


if __name__ == "__main__":
    main()
