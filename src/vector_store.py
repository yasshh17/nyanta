
from typing import List, Tuple
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()


class PineconeVectorStore:

    def __init__(self, index_name: str = "rag-chatbot"):
        self.index_name = index_name

        self.pinecone_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_key:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'PINECONE_API_KEY' in st.secrets:
                    self.pinecone_key = st.secrets['PINECONE_API_KEY']
            except:
                pass

        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                    openai_key = st.secrets['OPENAI_API_KEY']
            except:
                pass

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        self.vector_store = None
        print(f"✓ Connected to Pinecone index: {index_name}")

    def _get_raw_index(self):
        pc = Pinecone(api_key=self.pinecone_key)
        return pc.Index(self.index_name)

    def clear_index(self):
        index = self._get_raw_index()
        try:
            index.delete(delete_all=True)
        except Exception:
            pass
        self.vector_store = None

    def create_index(self, documents: List[Document]):
        print(f"Creating embeddings for {len(documents)} chunks...")
        self.vector_store = LangchainPinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print("✓ Documents indexed in Pinecone")

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            self.vector_store = LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results


if __name__ == "__main__":
    from document_loader import load_and_chunk

    chunks = load_and_chunk("data/sample.txt")
    store = PineconeVectorStore()
    store.create_index(chunks)

    results = store.search("What is machine learning?", k=2)
    print(f"\nFound {len(results)} results:")
    if results:
        doc, score = results[0]
        print(f"(score={score:.3f}) {doc.page_content[:200]}")
