# QuantAI Web Interface Guide

## 🌐 Interactive Web Application

QuantAI includes a beautiful, user-friendly web interface for managing your knowledge base without touching the command line.

---

## 🚀 Quick Start

### 1. Start the Web Server

```bash
# Method 1: Using Python directly
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Method 2: Using the module directly
cd api && python main.py

# Method 3: With auto-reload for development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Interface

Open your browser and navigate to:
```
http://localhost:8000
```

or

```
http://localhost:8000/static/index.html
```

---

## 🎨 Features

### 1. **📤 Document Upload**

**Drag & Drop Interface:**
- Drag PDF or TXT files directly into the upload zone
- Or click to browse and select files
- Multiple file upload supported
- Optional custom document naming

**Supported Formats:**
- ✅ **PDF files** (`.pdf`)
- ✅ **Text files** (`.txt`)
- 🔜 **EPUB, DOCX** (coming soon)

**What Happens:**
1. File is uploaded to server
2. Text is extracted (PDF → text)
3. Content is chunked (1000 chars + 200 overlap)
4. Knowledge is extracted (concepts, strategies, formulas)
5. Embeddings are generated (OpenAI or local)
6. Vectors are stored in ChromaDB
7. Success message shows how many chunks were added

**Example:**
- Upload: `Pairs_Trading_Strategy.pdf` (25 pages)
- Result: "✅ Successfully indexed: Pairs_Trading_Strategy (47 chunks)"
- Time: ~30-60 seconds

### 2. **🔍 Knowledge Search**

**Semantic Search:**
- Type your query in natural language
- Press Enter or click "Search"
- Get ranked results with relevance scores

**Results Display:**
- **Type badge:** Shows if it's a concept, strategy, formula, example, or code
- **Relevance score:** 0-1 score showing match quality
- **Source:** Which document it came from
- **Full content:** Complete text of the knowledge chunk

**Example Queries:**
```
"What is mean reversion?"
"Sharpe ratio calculation"
"Pairs trading cointegration test"
"High volatility strategies"
"Moving average crossover code"
```

**Search Features:**
- Fast: <100ms for most queries
- Semantic: Understands meaning, not just keywords
- Ranked: Best matches first
- Filtered: Can filter by type (future feature)

### 3. **🤖 AI Q&A (RAG-Powered)**

**Ask Complex Questions:**
- Type questions in natural language
- AI searches knowledge base for context
- GPT-4o generates comprehensive answers
- Automatic source citations

**What You Get:**
- **Your question** (displayed back)
- **AI answer** (comprehensive, contextual)
- **Sources used** (top 5 with relevance scores)
- **Confidence score** (0-1, higher = more confident)
- **Model info** (which GPT model was used)

**Example Questions:**
```
"How do I implement a pairs trading strategy with dynamic hedge ratios?"

"Compare momentum vs mean reversion strategies for forex markets"

"What's the formula for Sharpe ratio and when should I use it?"

"Explain cointegration testing with ADF test and provide code example"
```

**Answer Quality:**
- ✅ Expert-level responses
- ✅ Cites specific documents
- ✅ Combines multiple sources
- ✅ Includes formulas and examples
- ✅ Tailored to your knowledge base

### 4. **📊 Live Statistics Dashboard**

**Real-Time Metrics:**
- **Total Chunks:** How many knowledge pieces indexed
- **Vector Store:** Which database (ChromaDB)
- **Embedding Model:** Which model is being used
- **Status:** 🟢 Ready or 🔴 Error

**Auto-Refresh:**
- Updates every 10 seconds
- Shows immediate results after upload

---

## 🎯 Typical Workflow

### Daily Research Workflow

**Morning: Upload New Documents**
1. Open web interface
2. Drag 5 Bloomberg articles into upload zone
3. Wait 2-5 minutes for indexing
4. See "47 chunks added" success message

**During Day: Search & Query**
5. Search for "VIX volatility strategies"
6. Get 10 relevant results from your documents
7. Click to read full content

**Research Questions:**
8. Ask AI: "What strategies work best when VIX > 30?"
9. Get comprehensive answer with citations
10. Follow up: "Show me historical examples"
11. AI synthesizes from your documents

**Result:**
- Your proprietary knowledge at your fingertips
- AI-powered insights from YOUR data
- Fast, accurate, cited responses

---

## 🔧 API Endpoints

### Document Upload

**POST** `/api/upload`

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf" \
  -F "book_name=My Document"
```

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "book_name": "My Document",
  "chunks_added": 45,
  "message": "Successfully indexed 45 chunks from 'My Document'"
}
```

### Search Knowledge Base

**POST** `/api/search`

```bash
curl -X POST "http://localhost:8000/api/search" \
  -d "query=pairs trading" \
  -d "max_results=10"
```

**Response:**
```json
{
  "success": true,
  "query": "pairs trading",
  "num_results": 10,
  "results": [
    {
      "content": "Pairs trading is a market-neutral strategy...",
      "source_book": "Pairs Trading Guide",
      "chunk_type": "concept",
      "relevance_score": 0.892,
      "tags": ["pairs_trading", "arbitrage"]
    }
  ]
}
```

### Ask AI (RAG)

**POST** `/api/ask`

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -d "question=How do I calculate Sharpe ratio?" \
  -d "include_citations=true"
```

**Response:**
```json
{
  "success": true,
  "question": "How do I calculate Sharpe ratio?",
  "answer": "The Sharpe ratio is calculated as...",
  "sources": [
    {
      "book": "Quantitative Trading Basics",
      "type": "formula",
      "relevance": 0.945
    }
  ],
  "confidence": 0.923,
  "num_sources": 3,
  "model": "gpt-4o"
}
```

### Get Statistics

**GET** `/api/stats`

```bash
curl http://localhost:8000/api/stats
```

**Response:**
```json
{
  "total_chunks": 1247,
  "vector_store": "ChromaDB",
  "embedding_model": "all-MiniLM-L6-v2",
  "timestamp": "2025-11-13T19:30:00Z"
}
```

### Health Check

**GET** `/api/health`

```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "knowledge_engine": "initialized",
  "rag_pipeline": "available",
  "stats": {
    "total_chunks": 1247
  }
}
```

---

## 💻 Deployment Options

### Local Development

```bash
# Start with auto-reload
uvicorn api.main:app --reload --port 8000
```

### Production (Single Machine)

```bash
# Start with multiple workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Containerized)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t quantai .
docker run -p 8000:8000 -v $(pwd)/data:/app/data quantai
```

### Behind Nginx (Reverse Proxy)

```nginx
server {
    listen 80;
    server_name quantai.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://localhost:8000/api/;
    }

    client_max_body_size 100M;  # Allow large PDF uploads
}
```

### Systemd Service (Auto-start on Boot)

```ini
# /etc/systemd/system/quantai.service
[Unit]
Description=QuantAI Knowledge Manager
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/QuantAI
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable quantai
sudo systemctl start quantai
```

---

## 🔒 Security Considerations

### For Local/Internal Use

**Default Setup (No Auth):**
- ✅ Fine for local machine
- ✅ Fine for internal network
- ⚠️ NOT for public internet

**Recommendations:**
1. Firewall: Block port 8000 from external access
2. VPN: Access only via company VPN
3. Network: Use private IP ranges only

### For Multi-User/Production

**Add Authentication:**

```python
# api/main.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "secure_password":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials

@app.post("/api/upload", dependencies=[Depends(verify_credentials)])
async def upload_document(...):
    # Protected endpoint
```

**Use HTTPS:**

```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with HTTPS
uvicorn api.main:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

**Rate Limiting:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/ask")
@limiter.limit("10/minute")  # Max 10 queries per minute
async def ask_ai(...):
    ...
```

---

## 📱 Mobile-Friendly

The web interface is fully responsive and works on:
- ✅ Desktop browsers (Chrome, Firefox, Safari, Edge)
- ✅ Tablets (iPad, Android tablets)
- ✅ Mobile phones (iPhone, Android)

**Features on Mobile:**
- Touch-friendly upload zone
- Responsive layout
- Full functionality
- Fast loading

---

## 🎨 Customization

### Change Colors/Branding

Edit `api/static/index.html`:

```css
/* Change primary color */
background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);

/* Change header color */
.header h1 {
    color: #YOUR_BRAND_COLOR;
}
```

### Add Company Logo

```html
<div class="header">
    <img src="/static/logo.png" alt="Logo" style="height: 50px;">
    <h1>Your Company - QuantAI</h1>
</div>
```

### Custom Welcome Message

```javascript
// Add welcome popup on first visit
if (!localStorage.getItem('visited')) {
    alert('Welcome to QuantAI Knowledge Manager!');
    localStorage.setItem('visited', 'true');
}
```

---

## 🐛 Troubleshooting

### Issue: "Connection Refused"

**Solution:**
```bash
# Check if server is running
ps aux | grep uvicorn

# Start the server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Issue: "RAG Pipeline Not Available"

**Solution:**
```bash
# Add OpenAI API key to .env
echo "OPENAI_API_KEY=sk-your-key" >> .env

# Restart server
```

### Issue: "Upload Failed"

**Possible causes:**
1. File too large (increase `client_max_body_size` in nginx)
2. Wrong file format (only PDF/TXT supported)
3. Corrupted PDF (try different file)

**Check logs:**
```bash
# View server logs
tail -f /var/log/quantai.log
```

### Issue: "Slow Search"

**Solutions:**
1. Too many chunks? Optimize ChromaDB settings
2. Low RAM? Increase system memory
3. Use SSD for data storage
4. Enable GPU acceleration

---

## 📊 Performance Tips

### For Best Performance:

**Hardware:**
- SSD for data storage (3x faster than HDD)
- 16GB+ RAM for smooth operation
- Multiple CPU cores for parallel processing

**Configuration:**
```python
# api/main.py - Increase workers
uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

**Caching:**
```python
# Add Redis cache for frequent queries
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    FastAPICache.init(RedisBackend(...), prefix="quantai-cache")
```

---

## 🎯 Use Cases

### 1. Daily Research Assistant

**Workflow:**
1. Morning: Upload Bloomberg articles via web
2. Throughout day: Search for specific topics
3. Before trading: Ask AI about current strategies
4. Evening: Review and add research notes

### 2. Team Knowledge Base

**Setup:**
- Deploy on internal server
- Multiple researchers upload documents
- Shared knowledge base grows
- Everyone benefits from collective intelligence

### 3. Client Reporting

**Workflow:**
1. Upload client portfolio documents
2. Ask: "Summarize performance for Q3"
3. Get AI-generated summary with citations
4. Review and send to client

### 4. Compliance Monitoring

**Setup:**
- Index regulatory documents
- Search for specific rules
- Ask about compliance requirements
- Get instant answers with regulation citations

---

## 🚀 Next Steps

1. **Start the server**: `uvicorn api.main:app --reload`
2. **Open browser**: http://localhost:8000
3. **Upload documents**: Drag & drop your PDFs
4. **Start querying**: Search and ask questions
5. **Integrate**: Use API endpoints in your tools

**Your quantitative research just got a powerful web interface!** 🎉

---

## 📞 Support

For issues or questions:
1. Check logs: `tail -f api.log`
2. Review API docs: http://localhost:8000/docs
3. Test endpoints: http://localhost:8000/redoc

