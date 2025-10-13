"""Pinecone vector store with OpenAI embeddings."""

from typing import List
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()


class PineconeVectorStore:
    """Pinecone vector store with OpenAI embeddings."""
    
    def __init__(self, index_name: str = "rag-chatbot"):
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        print(f"✓ Connected to Pinecone index: {index_name}")
    
    def create_index(self, documents: List[Document]):
        """Create embeddings and store in Pinecone."""
        print(f"Creating embeddings for {len(documents)} chunks...")
        self.vector_store = LangchainPinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print("✓ Documents indexed in Pinecone")
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents."""
        if not self.vector_store:
            self.vector_store = LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
        results = self.vector_store.similarity_search(query, k=k)
        return results


if __name__ == "__main__":
    from document_loader import load_and_chunk
    
    chunks = load_and_chunk("data/sample.txt")
    store = PineconeVectorStore()
    store.create_index(chunks)
    
    results = store.search("What is machine learning?", k=2)
    print(f"\nFound {len(results)} results:")
    if results:
        print(results[0].page_content[:200])