<div align="center">

# Nyanta

### AI-Powered Personal Knowledge Assistant

Nyanta is an AI-powered personal knowledge assistant that turns documents into a conversational knowledge base using RAG (retrieval-augmented generation). Upload PDFs, DOCX, or TXT — Nyanta indexes them, answers natural-language queries, and returns cited source snippets.

**Intelligent document search with Retrieval-Augmented Generation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/_LangChain-0.3-green)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo](https://nyanta.streamlit.app)** • **[Documentation](#quick-start)** • **[Report Issue](https://github.com/yasshh17/nyanta/issues)**

![Nyanta Interface](docs/demo.png)

*Upload documents, ask questions, get answers with citations—all powered by AI*

</div>

---

## Overview

Nyanta is a production grade RAG chatbot that transforms static documents into conversational knowledge. Upload PDFs research papers, or notes, then query them in natural language with semantic search and source-cited answers.

**Built to solve:** Information overload. Manually searching through hundreds of pages takes hours. Keyword search misses context. Nyanta reduces document lookup time by **60-75%** using AI-powered semantic retrieval.

### How It Works
User Upload (PDF/TXT/DOCX)
   ↓
Document Loader → Text Chunking
   ↓
OpenAI Embeddings → Pinecone Vector Store
   ↓
User Query → Semantic Search → Top-K Context
   ↓
Groq LLM → Cited Answer

**RAG Pipeline:** Document → Chunks → Embeddings → Vector Search → LLM Generation → Cited Answer

---

## Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Understands meaning, not just keywords (vector similarity) |
| **Multi-Format** | PDF, TXT, DOCX support with intelligent parsing |
| **Advanced RAG** | Complete pipeline: chunking → embeddings → retrieval → generation |
| **Source Citations** | Every answer includes document references and chunk IDs |
| **Persistent Sessions** | Chat history survives page refreshes (SQLite) |
| **Real-Time Stats** | Live dashboard: vectors, documents, chunks |

**Technical Highlights:**
- Context-preserving semantic chunking (1000 tokens, 200 overlap)
- Hybrid storage: Pinecone (vectors) + SQLite (metadata)
- Session management with unique IDs
- Graceful error handling and retry logic
- Cost-optimized: $0.02 per 100 documents

---

## Tech Stack

**AI/ML**
- **LLM:** Groq (Llama 3.3 70B) - Free, fast inference
- **Embeddings:** OpenAI text-embedding-3-small (1536-dim)
- **Vector DB:** Pinecone Serverless (100K free vectors)
- **Framework:** LangChain 0.3 (RAG orchestration)

**Backend**
- **Runtime:** Python 3.11
- **UI:** Streamlit 1.39
- **Storage:** SQLite (chat history)
- **Parsing:** pypdf, python-docx

---

## Quick Start

### Prerequisites

# Required
Python 3.11+
Git

# API Keys (free signup)
Groq: https://console.groq.com
OpenAI: https://platform.openai.com ($5 minimum)
Pinecone: https://www.pinecone.io (free tier)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yasshh17/nyanta.git
cd nyanta

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EOF

# 5. Create foolproof startup script
cat > start_nyanta.sh << 'EOF'
#!/bin/bash

echo "Starting Nyanta..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3.11 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Verify correct Python
echo "✓ Using Python: $(which python)"
echo "✓ Python version: $(python --version)"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create it with API keys."
fi

echo "Launching application..."
echo ""

# Run application
python -m streamlit run app.py
EOF

# Make it executable
chmod +x start_nyanta.sh

# Test it works
./start_nyanta.sh

### First-Time Setup
Create Pinecone Index:

Go to Pinecone Console
Create index: rag-chatbot
Dimensions: 1536, Metric: cosine, Region: us-east-1

Get API Keys:

Groq: Console → API Keys → Create (free)
OpenAI: Platform → API Keys → Create ($5 credit needed)
Pinecone: API Keys → Copy default key

## Usage
**Basic Workflow**

Upload → Click "Browse files" in sidebar
Process → Click "Process Documents" (creates embeddings)
Query → Ask questions in natural language
Verify → Check sources in expandable citations

## Example Session
 Uploaded: research_paper.pdf (25 pages)
 Processed: 87 chunks in 30 seconds

 User: "What are the main findings about transformer architecture?"

Nyanta: "Based on the research paper, the main findings include:

1. Self-attention mechanisms enable parallel processing of sequences, 
   reducing training time by 80% vs RNNs (Source: research_paper.pdf, Chunk 23)

2. Multi-head attention captures different representation subspaces,
   improving model expressiveness (Source: research_paper.pdf, Chunk 31)

3. Positional encodings preserve sequence order without recurrence
   (Source: research_paper.pdf, Chunk 27)"

 *3 sources cited*

## Architecture
**System Components:**

┌─────────────────────────────────────────────────────┐
│                  STREAMLIT UI                        │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │   Upload   │  │    Chat     │  │  Statistics  │ │
│  └────────────┘  └─────────────┘  └──────────────┘ │
└────────────────────────┬────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Document   │  │  RAG Chain   │  │   Database   │
│    Loader    │  │              │  │   (SQLite)   │
└──────┬───────┘  └──────┬───────┘  └──────────────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│   Chunking   │  │   Retrieval  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────┐
│      OpenAI Embeddings API        │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│       Pinecone Vector DB          │
│  ┌─────────────────────────────┐ │
│  │  Vector Search Engine       │ │
│  │  • Cosine Similarity        │ │
│  │  • Top-K Retrieval          │ │
│  └─────────────────────────────┘ │
└──────────────────────────────────┘

**Data Flow:**  
Documents → Extraction → Semantic Chunks → OpenAI Embeddings → Pinecone  
Query → Cosine similarity → Top-K chunks → Groq LLM → Answer with citations → SQLite session persistence

**Key Design Decisions:**
1. **RAG vs Fine-Tuning:** Real-time knowledge updates, transparent citations, low cost  
2. **Cloud Vector DB (Pinecone):** Stateless deployment, scaling, production-ready  
3. **Groq for Generation:** 650–750 tokens/s, free tier, Llama 3.3 quality  
4. **SQLite Persistence:** Zero-config, ACID-compliant, upgradeable  


## 📁 Project Structure
nyanta/
├── app.py                    # Main Streamlit application
├── src/
│   ├── document_loader.py    # Document ingestion & chunking
│   ├── vector_store.py       # Pinecone vector operations
│   ├── rag_chain.py          # RAG pipeline & LLM integration
│   └── database.py           # SQLite persistence layer
│             
├── .streamlit/
│   └── config.toml          # Theme configuration
├── data/                    # Sample documents
├── documents/               # User uploads (gitignored)
├── .env                     # API keys (gitignored)
├── .env.example            # Environment template
├── .gitignore
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md

## Development
**Run Tests**

python src/document_loader.py  # Document processing
python src/vector_store.py     # Vector operations
python src/rag_chain.py        # RAG pipeline

**Run with sample data**
streamlit run app.py
# Upload files from data/ directory

## Environment Variables
**Required**

GROQ_API_KEY="gsk_your_key_here"           # Groq Console
OPENAI_API_KEY="sk_your_key_here"          # OpenAI Platform
PINECONE_API_KEY="pcsk_your_key_here"      # Pinecone Dashboard

**Optional**
CHUNK_SIZE=1000                # Default chunk size
CHUNK_OVERLAP=200              # Overlap between chunks

## Deployment
**Deploy to Streamlit Cloud (Free)**

### Step 1: Push to GitHub

git add .
git commit -m "Ready for deployment"
git push origin main 

### Step 2: Deploy on Streamlit Cloud

Go to share.streamlit.io
Click "New app"
Select repository: yasshh17/nyanta
Main file: app.py
Click "Advanced settings" 


### Step 3: Add secrets (API keys):

GROQ_API_KEY = "gsk_your_key_here"
OPENAI_API_KEY = "sk_your_key_here"
PINECONE_API_KEY = "pcsk_your_key_here" 

Click "Deploy"

Your app goes live at: https://yasshh17-nyanta.streamlit.app
Deployment time: ~2 minutes 

## Cost Analysis
**One-Time Setup**

OpenAI: $5 minimum credit
Groq: Free
Pinecone: Free (100K vectors)
Total: $5

**Per-Usage Costs**

Embeddings: $0.02 per 1M tokens (~100 documents)
LLM inference: $0 (Groq free tier)
Vector storage: $0 (Pinecone free tier)

Example: Processing 50 documents + 100 queries = ~$0.10

## Troubleshooting

**ModuleNotFoundError**
bashsource venv/bin/activate
pip install -r requirements.txt

**Pinecone index not found**
Create index in Pinecone dashboard
Verify index name: rag-chatbot
Check API key in .env

**OpenAI quota exceeded**
Add credits at platform.openai.com/billing
$5 minimum provides months of usage

**Slow processing**
Reduce CHUNK_SIZE in .env to 500
Process large PDFs in smaller batches

**Chat history not persisting**
Check chat_data.db file exists
Verify SQLite permissions
Restart Streamlit app


## Technical Achievements
What this project demonstrates:
✅ RAG Architecture - Production-grade retrieval pipeline
✅ Vector Databases - Embeddings, similarity search, indexing
✅ LLM Integration - Prompt engineering, context management
✅ Full-Stack Development - Backend + Frontend + Database
✅ Production Deployment - Live demo on cloud infrastructure
✅ System Design - Error handling, persistence, session management
Skills shown: Python, LangChain, Vector DBs, LLMs, SQL, Git, Cloud Deployment, API Integration

## Contributing
Contributions are welcome!  
Priority areas:
- Retrieval improvements (Hybrid search, reranking, HyDE)
- Evaluation frameworks (RAGAS, quality metrics)
- UI enhancements (themes, accessibility)
- Integrations (Google Drive, Notion, Confluence)

**Development workflow**
git checkout -b feature/your-feature

# Make changes and test locally
streamlit run app.py

# Commit and push changes
git commit -m "Add feature: description"
git push origin feature/your-feature

# Create a pull request on GitHub

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
**Yash Tambakhe** 
Building intelligent document tools and RAG systems.  
Open to roles in AI engineering and ML infrastructure.

- GitHub: https://github.com/yasshh17
- LinkedIn: https://www.linkedin.com/in/yash-tambakhe/
- Email: yashtambakhe@gmail.com

---

## Acknowledgments

- **LangChain** – RAG framework and documentation
- **Pinecone** – Vector database infrastructure
- **Groq** – High-speed inference platform
- **Streamlit** – Rapid UI development framework
- **OpenAI** – Embeddings and model APIs

---

⭐ If you find this project useful, [star this repo](https://github.com/yasshh17/nyanta)!

Built by **Yash Tambakhe**
