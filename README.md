# 📚 Semantic Search — Hands-On Large Language Models

A semantic search engine built on top of the **Hands-On Large Language Models** book by Jay Alammar. Upload the PDF once, then ask any question and get the most relevant passages back — powered by **Cohere Embeddings**, **ChromaDB**, and **FastAPI**.

---

## 🏗️ Architecture (MVC)

```
semantic_search/
│
├── main.py                        # Entry point — FastAPI + Gradio
│
├── models/
│   ├── embeddings_model.py        # Cohere embed_documents / embed_query
│   └── vector_store.py            # ChromaDB store & query
│
├── views/
│   └── gradio_view.py             # Gradio UI at /ui
│
├── controllers/
│   ├── ingest_controller.py       # PDF extraction + chunking
│   └── search_controller.py       # Search pipeline
│
├── routes/
│   ├── ingest_routes.py           # POST /api/v1/upload-pdf
│   └── search_routes.py           # POST /api/v1/search
│
├── utils/
│   └── pdf_parser.py              # PyMuPDF text extraction
│
├── data/
│   └── chroma_db/                 # Vector DB storage (auto-created)
│
├── config.py                      # Settings via pydantic-settings
├── .env                           # API keys (never commit this!)
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Tool | Role |
|------|------|
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **PyMuPDF** | PDF text extraction |
| **Cohere embed-v4.0** | Text embeddings |
| **ChromaDB** | Vector database |
| **Gradio** | Web UI |
| **Pydantic** | Settings & validation |

---

## 🚀 Getting Started

### 1. Clone & setup environment

```bash
conda create -n semantic-search python=3.11
conda activate semantic-search
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the root directory:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

Get your free API key from [cohere.com](https://cohere.com)

### 3. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```
```json
{ "status": "ok" }
```

---

### Upload PDF
```http
POST /api/v1/upload-pdf
Content-Type: multipart/form-data
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | PDF file to ingest |

**Response:**
```json
{
  "message": "PDF ingested successfully.",
  "filename": "book.pdf",
  "total_chunks": 1725
}
```

---

### Semantic Search
```http
POST /api/v1/search
Content-Type: application/json
```

```json
{
  "query": "What are transformer models?"
}
```

**Response:**
```json
{
  "query": "What are transformer models?",
  "total_results": 5,
  "results": [
    {
      "text": "The Transformer is a combination of stacked encoder and decoder blocks...",
      "page_number": 38,
      "source": "./data/temp/book.pdf",
      "score": 0.4882
    }
  ]
}
```

---

## 🌐 Gradio UI

After running the server, open your browser at:

```
http://localhost:8000/ui
```

Type any question about the book and get instant results.

---

## 🔄 How It Works

### Ingestion Pipeline

```
PDF File
  │
  ▼ PyMuPDF
Extract text per page
  │
  ▼ ingest_controller
Split into chunks (size=500, overlap=50)
  │
  ▼ Cohere embed-v4.0  [search_document]
Generate embedding vectors (batches of 90)
  │
  ▼ ChromaDB
Store vectors + text + metadata
```

### Search Pipeline

```
User Query
  │
  ▼ Cohere embed-v4.0  [search_query]
Generate query vector
  │
  ▼ ChromaDB
Cosine similarity search → Top 5 results
  │
  ▼ API Response
text + page_number + score
```

---

## ⚙️ Configuration

All settings are in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `COHERE_API_KEY` | required | Your Cohere API key |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of search results |
| `CHROMA_COLLECTION_NAME` | `llm_book` | ChromaDB collection name |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage path |

---

## 📖 Interactive API Docs

FastAPI auto-generates Swagger docs at:

```
http://localhost:8000/docs
```

---

## 📝 Notes

- Cohere free tier allows **100 requests/month** on the embed endpoint
- ChromaDB stores data locally in `./data/chroma_db/` — no external service needed
- The PDF is deleted from temp storage after ingestion
- Re-uploading the same PDF will add duplicate chunks — clear `./data/chroma_db/` first if needed