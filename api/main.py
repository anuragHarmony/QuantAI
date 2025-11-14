"""
FastAPI application for QuantAI platform
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional
from loguru import logger
import uvicorn

from shared.config.settings import settings
from ai_agent.tools.base import global_registry, ToolResult
from ai_agent.tools.trading_tools import (
    GetMarketDataTool,
    CalculateIndicatorTool,
    GenerateSignalsTool,
    CalculateMetricsTool
)


# API Models
class ToolExecuteRequest(BaseModel):
    """Request to execute a tool"""
    tool_name: str = Field(description="Name of tool to execute")
    parameters: dict[str, Any] = Field(description="Tool parameters")


class ChatRequest(BaseModel):
    """Chat request"""
    message: str = Field(description="User message")
    context: Optional[dict[str, Any]] = Field(default=None, description="Additional context")
    use_tools: bool = Field(default=True, description="Whether to use tools")


class ChatResponse(BaseModel):
    """Chat response"""
    message: str = Field(description="AI response")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tools that were called")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalRequest(BaseModel):
    """Knowledge retrieval request"""
    query: str = Field(description="Search query")
    top_k: int = Field(default=10, description="Number of results")
    filters: Optional[dict[str, Any]] = Field(default=None, description="Metadata filters")


class DocumentIngestRequest(BaseModel):
    """Document ingestion request"""
    file_path: Optional[str] = Field(default=None, description="Local file path")
    url: Optional[str] = Field(default=None, description="URL to fetch document from")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Document metadata")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-Human Collaborative Quant Research Platform"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting QuantAI API server...")

    # Register trading tools
    global_registry.register(GetMarketDataTool())
    global_registry.register(CalculateIndicatorTool())
    global_registry.register(GenerateSignalsTool())
    global_registry.register(CalculateMetricsTool())

    logger.info(f"Registered {len(global_registry.get_all_tools())} tools")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down QuantAI API server...")


# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.environment
    }


# Tool endpoints
@app.get("/tools")
async def list_tools():
    """List all available tools"""
    tools = global_registry.get_all_tools()

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type.value,
                        "description": p.description,
                        "required": p.required
                    }
                    for p in tool.get_parameters()
                ]
            }
            for tool in tools
        ]
    }


@app.get("/tools/{tool_name}")
async def get_tool(tool_name: str):
    """Get tool definition"""
    tool = global_registry.get_tool(tool_name)

    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    definition = tool.get_definition()

    return {
        "name": definition.name,
        "description": definition.description,
        "category": definition.category,
        "parameters": [p.dict() for p in definition.parameters],
        "openai_function": definition.to_openai_function(),
        "anthropic_tool": definition.to_anthropic_tool()
    }


@app.post("/tools/execute")
async def execute_tool(request: ToolExecuteRequest):
    """Execute a tool"""
    logger.info(f"Executing tool: {request.tool_name}")

    result = await global_registry.execute_tool(
        request.tool_name,
        **request.parameters
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "success": result.success,
        "result": result.result,
        "metadata": result.metadata
    }


# Chat endpoint (simplified version)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with AI assistant

    This is a simplified version. In production, this would use the full RAG pipeline.
    """
    try:
        from ai_agent.reasoner.llm_providers import LLMProviderFactory

        # Create LLM provider
        llm = LLMProviderFactory.create_default()

        # If tools are enabled, get available tools
        tool_calls = []
        if request.use_tools:
            tools = global_registry.get_openai_functions()

            if tools:
                # Use function calling
                response, function_call = await llm.complete_with_functions(
                    request.message,
                    tools
                )

                if function_call:
                    # Execute the function
                    tool_result = await global_registry.execute_tool(
                        function_call["name"],
                        **function_call["arguments"]
                    )

                    tool_calls.append({
                        "tool": function_call["name"],
                        "arguments": function_call["arguments"],
                        "result": tool_result.result
                    })

                    # Generate final response with tool result
                    final_prompt = f"""
Original query: {request.message}

Tool called: {function_call['name']}
Tool result: {tool_result.result}

Please provide a helpful response to the user based on this information.
"""
                    response = await llm.complete(final_prompt)

            else:
                # No tools, just complete
                response = await llm.complete(request.message)
        else:
            # Tools disabled, just complete
            response = await llm.complete(request.message)

        return ChatResponse(
            message=response,
            tool_calls=tool_calls,
            metadata={
                "tools_used": len(tool_calls)
            }
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge retrieval endpoints
@app.post("/knowledge/retrieve")
async def retrieve_knowledge(request: RetrievalRequest):
    """Retrieve knowledge chunks"""
    try:
        # This would use the full RAG pipeline
        # For now, return a placeholder
        return {
            "query": request.query,
            "results": [],
            "metadata": {
                "message": "RAG pipeline not fully initialized. Add knowledge first."
            }
        }

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/ingest")
async def ingest_document(request: DocumentIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a document into knowledge base

    This is a placeholder for the full ingestion pipeline
    """
    if not request.file_path and not request.url:
        raise HTTPException(
            status_code=400,
            detail="Either file_path or url must be provided"
        )

    # In production, this would:
    # 1. Download/load document
    # 2. Process and extract text
    # 3. Chunk the content
    # 4. Generate embeddings
    # 5. Store in vector store and knowledge graph

    return {
        "status": "queued",
        "message": "Document ingestion queued for processing",
        "metadata": request.metadata or {}
    }


# Market data endpoints
@app.get("/market/data/{symbol}")
async def get_market_data(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d"
):
    """Get historical market data for a symbol"""
    tool = GetMarketDataTool()

    result = await tool.execute(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    return result.result


# Strategy testing endpoints
@app.post("/strategy/test")
async def test_strategy(
    strategy_type: str,
    symbol: str,
    start_date: str,
    end_date: str
):
    """
    Test a strategy on historical data

    This is a simplified version that chains tools together
    """
    try:
        # 1. Get market data
        data_tool = GetMarketDataTool()
        data_result = await data_tool.execute(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if not data_result.success:
            raise HTTPException(status_code=400, detail=data_result.error)

        prices = data_result.result.data["close"]

        # 2. Generate signals
        signal_tool = GenerateSignalsTool()
        signal_result = await signal_tool.execute(
            strategy=strategy_type,
            prices=prices
        )

        if not signal_result.success:
            raise HTTPException(status_code=400, detail=signal_result.error)

        # 3. Calculate simple returns (placeholder for full backtest)
        signals = signal_result.result["signals"]
        returns = []

        for i in range(1, len(prices)):
            if signals[i-1] == 1:  # Was long
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            elif signals[i-1] == -1:  # Was short
                ret = (prices[i-1] - prices[i]) / prices[i-1]
                returns.append(ret)

        # 4. Calculate metrics
        if returns:
            metrics_tool = CalculateMetricsTool()
            metrics_result = await metrics_tool.execute(returns=returns)

            if metrics_result.success:
                return {
                    "symbol": symbol,
                    "strategy": strategy_type,
                    "period": {"start": start_date, "end": end_date},
                    "signals": signal_result.result,
                    "metrics": metrics_result.result
                }

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "error": "Insufficient data for metrics"
        }

    except Exception as e:
        logger.error(f"Strategy test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
def start_server():
    """Start the API server"""
    uvicorn.run(
        "api.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        reload=settings.api.api_reload,
        workers=settings.api.api_workers
    )


if __name__ == "__main__":
    start_server()
