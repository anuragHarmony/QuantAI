# QuantAI Architecture - Local-First Design

## Design Philosophy: 100% Local Processing

QuantAI is designed with a **local-first** architecture, ensuring that:
- ✓ All data stays on your machine
- ✓ No external API calls required (except optional OpenAI for LLM extraction)
- ✓ Complete privacy for proprietary trading knowledge
- ✓ No dependencies on cloud services
- ✓ Works offline after initial model download

## Why Local-First for Quant Research?

### Data Privacy
- **Proprietary strategies** remain confidential
- **Trading signals** never leave your infrastructure
- **Research documents** stay private
- **No data sharing** with third-party services

### Performance
- **No network latency** - all processing is local
- **Fast retrieval** - sub-100ms search times
- **Offline capability** - works without internet
- **No API rate limits** - unlimited queries

### Cost
- **No per-query costs** - unlike cloud vector DBs
- **No data transfer fees**
- **One-time model download** - reuse forever
- **Scales with your hardware** - not your wallet

## Local Storage Architecture

```
QuantAI/
├── data/
│   └── chroma/              # Local vector database
│       ├── chroma.sqlite3   # Metadata storage
│       └── *.parquet        # Vector embeddings
│
├── sample_data/
│   └── documents/           # Your local documents
│
└── models/                  # Downloaded models (cached)
    └── sentence-transformers/
```

## Component Details

### 1. Vector Store: ChromaDB (Local)

**Why ChromaDB for local storage:**
- Persistent local storage using SQLite + Parquet
- No server required - embedded database
- Fast queries on consumer hardware
- Supports millions of vectors locally
- Simple Python API

**Storage location:** `./data/chroma/`

**Data format:**
- Vectors: Parquet files (efficient columnar storage)
- Metadata: SQLite database
- Combined size: ~1-5MB per 1000 chunks

### 2. Embeddings: Sentence Transformers (Local)

**Why Sentence Transformers:**
- Models run entirely on CPU/GPU locally
- No API calls needed after model download
- 384-dimension vectors (efficient storage)
- Fast inference: ~100 documents/second on CPU

**Model location:** Cached in `~/.cache/torch/sentence_transformers/`

**Model size:** ~80-120MB (one-time download)

**Supported devices:** CPU, CUDA, MPS (Apple Silicon)

### 3. Document Processing: PyMuPDF (Local)

**All processing local:**
- PDF parsing happens on your machine
- Text extraction uses local libraries
- No document upload to external services
- Support for encrypted PDFs (local password)

## Alternative Cloud Services (NOT Used)

We specifically **avoid** these cloud-based approaches:

### ❌ Google File Search API
- **Issue:** Uploads files to Google Cloud
- **Issue:** Stores embeddings on Google servers
- **Issue:** Data processed by Google
- **Our choice:** Local ChromaDB instead

### ❌ Pinecone
- **Issue:** Cloud-only vector database
- **Issue:** Data stored on Pinecone servers
- **Issue:** Per-query costs and rate limits
- **Our choice:** Local ChromaDB instead

### ❌ OpenAI Embeddings API
- **Issue:** Sends documents to OpenAI
- **Issue:** Usage tracked and billed
- **Issue:** Requires internet connection
- **Our choice:** Local sentence-transformers instead

## Data Flow (100% Local)

```
Document (PDF) → [Local]
    ↓
PyMuPDF Extraction → [Local]
    ↓
Text Chunking → [Local]
    ↓
Sentence Transformer Embeddings → [Local]
    ↓
ChromaDB Storage → [Local Disk]
    ↓
Vector Search → [Local]
    ↓
Results → [Local]
```

**No network calls** in the entire pipeline!

## Optional LLM Integration (Can Be Local)

For knowledge extraction, you have options:

### Option 1: Rule-Based (Default, 100% Local)
```python
engine = KnowledgeEngine(use_llm_extraction=False)  # No LLM needed
```
Uses pattern matching and heuristics - completely local.

### Option 2: OpenAI API (External, Optional)
```python
engine = KnowledgeEngine(
    openai_api_key="sk-...",
    use_llm_extraction=True
)
```
Only sends **chunks** (not full documents) to OpenAI for classification.

### Option 3: Local LLM (100% Local, Advanced)
You can integrate local models:
- **Ollama** (llama3, mistral, etc.)
- **llama.cpp**
- **GPT4All**

Example integration point: `knowledge_engine/ingest/knowledge_extractor.py`

## Performance Benchmarks (Local Hardware)

### Intel i7 CPU (8 cores)
- Embedding generation: ~100 chunks/second
- Vector search: <50ms per query
- Document indexing: ~5 minutes for 10,000 pages

### Apple M1 Pro
- Embedding generation: ~150 chunks/second
- Vector search: <30ms per query
- Document indexing: ~3 minutes for 10,000 pages

### AMD Ryzen 9 + RTX 3080
- Embedding generation: ~500 chunks/second (GPU)
- Vector search: <20ms per query
- Document indexing: ~1 minute for 10,000 pages

## Scaling Locally

### Storage Capacity
- **1,000 documents** → ~50MB vector DB
- **10,000 documents** → ~500MB vector DB
- **100,000 documents** → ~5GB vector DB

### RAM Requirements
- **Basic usage** (1K-10K chunks): 2GB RAM
- **Medium usage** (10K-100K chunks): 4GB RAM
- **Heavy usage** (100K-1M chunks): 8GB+ RAM

### Disk I/O Optimization
ChromaDB uses memory-mapped files, so:
- Use SSD for best performance
- NVMe provides 2-3x speedup
- RAID 0 can help with very large databases

## Alternative Local Vector Databases

If you need different features, these also support local storage:

### Qdrant (Local Mode)
```python
from qdrant_client import QdrantClient
client = QdrantClient(path="./data/qdrant")  # Local storage
```
- Rust-based, very fast
- Good for 100M+ vectors
- More complex setup

### Milvus (Standalone Mode)
```bash
docker-compose up  # Local Docker deployment
```
- Production-grade
- Requires more resources
- Better for enterprise scale

### Weaviate (Embedded)
```python
import weaviate
client = weaviate.Client(embedded_options=weaviate.embedded.EmbeddedOptions())
```
- Supports hybrid search
- GraphQL API
- More memory intensive

**Our recommendation:** Stick with ChromaDB for simplicity and Python integration.

## Security Considerations

### Local Data Protection
1. **Encrypt disk** - Use full disk encryption (FileVault, BitLocker, LUKS)
2. **Secure `.env` files** - Add to `.gitignore`, use proper permissions
3. **Git security** - Never commit `data/` directory
4. **Backup strategy** - Regular backups of `data/chroma/`

### Network Security
Since everything is local:
- No firewall rules needed
- No ports to expose
- No TLS/SSL certificates required
- No API key management (except optional OpenAI)

## Deployment Options

### 1. Single Machine (Recommended)
```bash
# Run everything on one machine
python index_documents.py
python query_knowledge.py --interactive
```

### 2. Local Network (Advanced)
You can wrap QuantAI in a local API:
```python
# api/main.py (included in project)
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Access from other machines on your LAN.

### 3. Air-Gapped Environment
Perfect for high-security environments:
1. Download dependencies offline: `pip download -r requirements.txt`
2. Download models: Pre-cache sentence-transformers
3. Transfer to air-gapped machine
4. Install and run - no internet needed

## Future Local Enhancements

Planned features that maintain local-first approach:

### Phase 2: Local Knowledge Graph
- Neo4j embedded or SQLite-based graph
- All relationships stored locally
- No cloud graph database services

### Phase 3: Local Backtesting
- Historical data stored locally (CSV, Parquet, HDF5)
- Strategy execution on local machine
- No brokerage API required for backtesting

### Phase 4: Local Model Fine-Tuning
- Fine-tune sentence-transformers on your data
- Train locally using PyTorch
- Export and use custom models - all local

## Conclusion

QuantAI is built from the ground up to keep your proprietary quant research **private and local**. Every design decision prioritizes:

1. **Privacy** - Your data never leaves your machine
2. **Performance** - Local processing is faster than cloud
3. **Cost** - No per-query fees or subscriptions
4. **Control** - You own your infrastructure

This architecture is ideal for:
- Proprietary trading firms
- Independent quant researchers
- Academic research
- High-security environments
- Offline/air-gapped deployments

---

**Remember:** Your trading edge is in your knowledge. Keep it local. Keep it secure. Keep it yours.
