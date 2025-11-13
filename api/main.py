"""FastAPI web application for QuantAI."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Optional
import shutil
import tempfile
from datetime import datetime
from loguru import logger

from knowledge_engine import KnowledgeEngine
from ai_agent.reasoner import RAGPipeline
from shared.config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Knowledge Manager",
    description="AI-powered knowledge management for quantitative finance research",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize knowledge engine and RAG pipeline
knowledge_engine = None
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize knowledge engine on startup."""
    global knowledge_engine, rag_pipeline

    logger.info("Initializing QuantAI Knowledge Engine...")
    try:
        knowledge_engine = KnowledgeEngine(use_llm_extraction=False)

        # Initialize RAG pipeline if OpenAI key is available
        if settings.openai_api_key:
            rag_pipeline = RAGPipeline(knowledge_engine=knowledge_engine)
            logger.info("RAG Pipeline initialized")
        else:
            logger.warning("No OpenAI API key - RAG will not be available")

        logger.info("QuantAI initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing QuantAI: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <html>
        <head><title>QuantAI</title></head>
        <body>
            <h1>QuantAI Knowledge Manager</h1>
            <p>API is running. Access the UI at /static/index.html</p>
        </body>
    </html>
    """


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    stats = knowledge_engine.get_stats() if knowledge_engine else {}
    return {
        "status": "healthy",
        "knowledge_engine": "initialized" if knowledge_engine else "not initialized",
        "rag_pipeline": "available" if rag_pipeline else "not available",
        "stats": stats
    }


@app.get("/api/stats")
async def get_stats():
    """Get knowledge base statistics."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge engine not initialized")

    stats = knowledge_engine.get_stats()
    return {
        "total_chunks": stats.get("total_chunks", 0),
        "vector_store": stats.get("vector_store", "Unknown"),
        "embedding_model": stats.get("embedding_model", "Unknown"),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    book_name: Optional[str] = Form(None)
):
    """
    Upload and index a document.

    Supports: PDF, TXT files
    """
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge engine not initialized")

    # Validate file type
    allowed_extensions = {".pdf", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    # Use provided name or filename
    book_name = book_name or Path(file.filename).stem

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)

        logger.info(f"Processing uploaded file: {file.filename} as '{book_name}'")

        # Index the document
        if file_ext == ".pdf":
            num_chunks = knowledge_engine.ingest_document(
                tmp_path,
                book_name=book_name
            )
        else:  # .txt
            # Read text file
            content = tmp_path.read_text(encoding='utf-8')

            # Process using document processor
            chunks = knowledge_engine.document_processor.chunk_text(content)

            # Extract knowledge
            all_knowledge_chunks = []
            for i, text_chunk in enumerate(chunks):
                knowledge_chunks = knowledge_engine.knowledge_extractor.extract_knowledge_chunks(
                    text=text_chunk,
                    source_book=book_name,
                    source_chapter=f"Section {i+1}"
                )
                all_knowledge_chunks.extend(knowledge_chunks)

            # Add to vector store
            if all_knowledge_chunks:
                knowledge_engine.vector_store.add_chunks(all_knowledge_chunks)

            num_chunks = len(all_knowledge_chunks)

        # Clean up temp file
        tmp_path.unlink()

        logger.info(f"Successfully indexed '{book_name}': {num_chunks} chunks")

        return {
            "success": True,
            "filename": file.filename,
            "book_name": book_name,
            "chunks_added": num_chunks,
            "message": f"Successfully indexed {num_chunks} chunks from '{book_name}'"
        }

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure temp file is cleaned up
        if 'tmp_path' in locals() and tmp_path.exists():
            tmp_path.unlink()


@app.post("/api/search")
async def search(
    query: str = Form(...),
    max_results: int = Form(10),
    chunk_type: Optional[str] = Form(None)
):
    """
    Search the knowledge base.

    Args:
        query: Search query
        max_results: Maximum number of results (default: 10)
        chunk_type: Filter by type (concept, strategy, formula, example, code)
    """
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge engine not initialized")

    try:
        # Build filters
        filters = {}
        if chunk_type:
            filters["chunk_type"] = chunk_type

        # Perform search
        results = knowledge_engine.search(
            query=query,
            max_results=max_results,
            filters=filters if filters else None
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.chunk.content,
                "source_book": result.chunk.source_book,
                "source_chapter": result.chunk.source_chapter,
                "chunk_type": result.chunk.chunk_type,
                "relevance_score": result.relevance_score,
                "tags": result.chunk.tags,
                "asset_class": result.chunk.asset_class,
                "strategy_type": result.chunk.strategy_type
            })

        return {
            "success": True,
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_ai(
    question: str = Form(...),
    include_examples: bool = Form(True),
    include_citations: bool = Form(True)
):
    """
    Ask a question and get AI-generated answer with RAG.

    Requires OpenAI API key to be configured.
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available. Please configure OPENAI_API_KEY in .env"
        )

    try:
        # Get AI answer
        result = rag_pipeline.ask(
            question=question,
            include_examples=include_examples,
            include_citations=include_citations
        )

        return {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "num_sources": result["num_sources"],
            "model": result["model"]
        }

    except Exception as e:
        logger.error(f"Error in AI ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(
    messages: List[dict]
):
    """
    Multi-turn conversation with AI.

    Args:
        messages: List of message dicts with 'role' and 'content'
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available. Please configure OPENAI_API_KEY in .env"
        )

    try:
        result = rag_pipeline.chat(messages, include_retrieval=True)

        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "model": result["model"]
        }

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/reset")
async def reset_knowledge_base():
    """Reset the entire knowledge base (WARNING: deletes all data)."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge engine not initialized")

    try:
        knowledge_engine.reset()
        return {
            "success": True,
            "message": "Knowledge base has been reset"
        }
    except Exception as e:
        logger.error(f"Error resetting knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
