"""Load and chunk documents for RAG."""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain.schema import Document
import os


def load_document(file_path: str) -> List[Document]:
    """Load a document based on file type."""
    _, ext = os.path.splitext(file_path)
    
    if ext == '.txt':
        loader = TextLoader(file_path)
    elif ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return loader.load()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        
    return chunks


def load_and_chunk(file_path: str) -> List[Document]:
    """Load and chunk a document in one step."""
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    print(f"✓ Loaded {file_path}")
    print(f"✓ Created {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    # Test
    chunks = load_and_chunk("data/sample.txt")
    print(f"\nFirst chunk preview:")
    print(chunks[0].page_content[:200])