"""
Document loading and chunking module

Chunking Strategy (Policy-Aware):
- HR and compensation documents contain structured rules, bullet lists, and exceptions
- We chunk primarily at paragraph / section level to preserve full policy meaning
- Larger chunk size is used to keep eligibility rules, conditions, and exceptions together
- Light overlap is added to maintain continuity across sections

"""

import os
from typing import List, Dict, Any


class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Maximum characters per chunk (larger for policy coherence)
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load all .txt documents from directory"""
        documents = []

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                documents.append({
                    "source": filename,
                    "content": content,
                    "metadata": {"source": filename}
                })

        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into meaningful policy chunks.

        Strategy:
        - First split by double newlines (paragraphs / sections)
        - Merge paragraphs until chunk_size is reached
        - Apply overlap between chunks
        """
        all_chunks = []

        for doc in documents:
            source = doc["source"]
            text = doc["content"]

            # Split into paragraphs/sections
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            current_chunk = ""
            chunk_index = 0

            for para in paragraphs:
                # If adding paragraph exceeds chunk size, finalize current chunk
                if len(current_chunk) + len(para) > self.chunk_size:
                    all_chunks.append({
                        "id": f"{source}_chunk_{chunk_index}",
                        "text": current_chunk.strip(),
                        "source": source,
                        "chunk_index": chunk_index,
                        "metadata": {
                            "source": source,
                            "chunk_index": chunk_index
                        }
                    })
                    chunk_index += 1

                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            # Add last chunk
            if current_chunk.strip():
                all_chunks.append({
                    "id": f"{source}_chunk_{chunk_index}",
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_index": chunk_index,
                    "metadata": {
                        "source": source,
                        "chunk_index": chunk_index
                    }
                })

        return all_chunks
