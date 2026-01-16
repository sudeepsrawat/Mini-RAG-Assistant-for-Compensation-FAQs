"""
Main RAG System for Compensation FAQs - Using gemini-flash-latest
Uses local embeddings (Sentence Transformers)
Uses Gemini for grounded answer generation
"""

import os
from typing import List, Dict, Any, Tuple
from google import genai
from dotenv import load_dotenv

load_dotenv()

from document_loader import DocumentLoader
from embeddings import EmbeddingGenerator, VectorStore


class RAGSystem:
    def __init__(self, documents_dir: str = "data", use_gemini: bool = True):
        self.documents_dir = documents_dir
        self.use_gemini = use_gemini

        self.loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_generator)

        self.load_and_index_documents()

        if self.use_gemini:
            self.initialize_gemini()

    # ------------------ GEMINI INIT ------------------

    def initialize_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ GEMINI_API_KEY missing. Gemini disabled.")
            self.use_gemini = False
            return

        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-flash-latest"

            test = self.client.models.generate_content(
                model=self.model_name,
                contents="Say hello"
            )

            if test and test.text:
                print("✅ Gemini API initialized successfully")
                self.gemini_available = True
            else:
                raise RuntimeError("Gemini test failed")

        except Exception as e:
            print(f"❌ Gemini init failed: {e}")
            self.use_gemini = False
            self.gemini_available = False

    # ------------------ DOCUMENT LOADING ------------------

    def load_and_index_documents(self):
        print(f"📂 Loading documents from {self.documents_dir}...")
        docs = self.loader.load_documents(self.documents_dir)
        print(f"   ✅ Loaded {len(docs)} documents")

        print("   🔄 Chunking documents...")
        chunks = self.loader.chunk_documents(docs)
        print(f"   ✅ Created {len(chunks)} chunks")

        self.vector_store.add_documents(chunks)
        print("   ✅ Documents indexed successfully")

    # ------------------ RETRIEVAL ------------------

    def retrieve_context(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        results = self.vector_store.search(query, k=k)

        print(f"\n🔍 Retrieved Context for: '{query}'")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n   Chunk {i} (Relevance: {score:.1%})")
            print(f"   📄 Source: {chunk['source']}")
            preview = chunk['text'][:250] + ("..." if len(chunk['text']) > 250 else "")
            print(f"   📝 Text: {preview}")

        # ✅ RETURN TOP RESULTS DIRECTLY (NO KEYWORD FILTERING)
        return results[:3]

    # ------------------ PROMPT ------------------

    def generate_prompt(self, query: str, context_chunks: List[Tuple[dict, float]]) -> str:
        merged_chunks = {}
        for chunk, _ in context_chunks:
            merged_chunks.setdefault(chunk["source"], "")
            merged_chunks[chunk["source"]] += chunk["text"] + "\n\n"

        context_text = ""
        for source, text in merged_chunks.items():
            context_text += f"Source: {source}\n{text}\n\n"

        return f"""
You are an HR compensation policy assistant.

RULES (MANDATORY):
- Use ONLY the information present in the context
- Do NOT use external knowledge or assumptions
- You MAY synthesize across multiple context sections
- If the context does not clearly answer the question,
  respond exactly with:
  "The information is not available in the provided documents."

CONTEXT:
----------------
{context_text}
----------------

QUESTION:
{query}

ANSWER (grounded strictly in context):
- Provide a complete, structured answer
- Do not stop mid-sentence
- List all applicable criteria found in the context
"""

    # ------------------ GENERATION ------------------

    def generate_with_gemini(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 800,
                }
            )

            if response.text:
                return response.text.strip()

            return "The information is not available in the provided documents."

        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return "The information is not available in the provided documents."

    # ------------------ ASK ------------------

    def ask(self, question: str) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"❓ QUESTION: {question}")
        print("=" * 60)

        retrieved = self.retrieve_context(question)

        # ✅ LOWERED THRESHOLD
        relevant_chunks = [(c, s) for c, s in retrieved if s >= 0.30]

        if not relevant_chunks:
            answer = "The information is not available in the provided documents."
            sources = []
        else:
            prompt = self.generate_prompt(question, relevant_chunks)
            answer = self.generate_with_gemini(prompt)
            sources = sorted({c["source"] for c, _ in relevant_chunks})

        print("\n" + "=" * 60)
        print("✅ ANSWER:")
        print("=" * 60)
        print(answer)

        if sources:
            print(f"\n📚 Sources: {', '.join(sources)}")

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


# ------------------ CLI ------------------

def main():
    import argparse

    parser = argparse.ArgumentParser("RAG Assistant for Compensation FAQs")
    parser.add_argument("--no-gemini", action="store_true")
    args = parser.parse_args()

    print("🚀 Initializing RAG System for Compensation FAQs...")
    rag = RAGSystem(use_gemini=not args.no_gemini)

    questions = [
        "How is bonus eligibility determined?",
        "What happens during a mid year role change?",
        "Are promotions tied to performance ratings?",
        "How often are salaries revised?",
        "What factors affect variable pay?",
    ]

    print("\n📋 Sample Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    while True:
        try:
            q = input("\n🤔 Enter question or number (exit to quit): ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q.isdigit() and 1 <= int(q) <= len(questions):
                rag.ask(questions[int(q) - 1])
            else:
                rag.ask(q)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
