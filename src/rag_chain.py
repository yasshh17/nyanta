"""RAG chain for question answering."""

from typing import List, Dict
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()


class RAGChain:
    """RAG chain for answering questions with citations."""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer that"
3. Cite your sources by mentioning the relevant parts of the context
4. Be concise but complete

Context:
{context}
"""),
            ("human", "{question}")
        ])
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'N/A')
            context_parts.append(
                f"[Source {i}: {source}, Chunk {chunk_id}]\n{doc.page_content}\n"
            )
        return "\n".join(context_parts)
    
    def query(self, question: str, documents: List[Document]) -> Dict:
        """Answer a question using retrieved documents."""
        context = self.format_context(documents)
        
        # Generate answer
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in documents
            ]
        }


if __name__ == "__main__":
    # Test
    from document_loader import load_and_chunk
    from vector_store import PineconeVectorStore
    
    # Load and index
    chunks = load_and_chunk("data/sample.txt")
    store = PineconeVectorStore()
    
    # Create index (Pinecone is cloud-based, no save/load needed)
    store.create_index(chunks)
    
    # Query
    rag = RAGChain()
    question = "What are the types of AI?"
    
    results = store.search(question, k=2)
    response = rag.query(question, results)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(response['sources'], 1):
        print(f"{i}. {source['source']} (Chunk {source['chunk_id']})")