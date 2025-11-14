from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from orchestrator import CacheOrchestrator
import uvicorn


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    llm_provider: Optional[str] = Field(None, description="LLM provider to use (openai or gemini)")


class QueryResponse(BaseModel):
    query: str
    response: str
    cache_layer: Optional[str]
    cache_hit: bool
    elapsed_time: float
    llm_called: bool
    llm_provider: Optional[str] = None
    rag_documents: Optional[int] = None


class DocumentRequest(BaseModel):
    content: str = Field(..., description="Document content to add to RAG cache")
    metadata: Optional[Dict] = Field(None, description="Optional metadata for the document")


class DocumentBatchRequest(BaseModel):
    documents: List[Dict] = Field(..., description="List of documents to add")


class HealthResponse(BaseModel):
    status: str
    components: Dict


# Initialize FastAPI app 
app = FastAPI(
    title="Agentic Cache-Driven Application",
    description="Multi-layer caching system with LLM integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = None

"""
@app.on_event("startup")
async def startup_event():
    #Initalize the cache orchestrator on startup
    global orchestrator
    try:
        orchestrator = CacheOrchestrator()
        print("✓ Application started successfully")
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        raise
"""

@app.on_event("startup")
async def startup_event():
    """Initialize the cache orchestrator on startup"""
    global orchestrator
    try:
        orchestrator = CacheOrchestrator()
        
        # Register dummy LLM provider for testing
        def dummy_llm(query: str, context: Optional[str] = None) -> str:
            """Dummy LLM for testing without API tokens"""
            query_lower = query.lower()
            
            if "machine learning" in query_lower:
                return "Machine learning is a subset of AI..."
            elif "python" in query_lower:
                return "Python is a high-level programming language..."
            else:
                return f"Dummy response for: '{query}'"
        
        orchestrator.llm_manager.register_custom_provider(
            provider_name="dummy",
            custom_function=dummy_llm,
            model_name="Dummy LLM for Testing"
        )
        
        print("✓ Dummy LLM provider registered")
        print("✓ Application started successfully")
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Agentic Cache-Driven Application API",
        "version": "1.0.0",
        "cache_layers": {
            "layer_0": "Exact Cache (Redis)",
            "layer_1": "Semantic Cache (Qdrant)",
            "layer_2": "RAG/Document Cache (Qdrant)"
        },
        "endpoints": {
            "query": "/api/query",
            "documents": "/api/documents",
            "health": "/api/health",
            "cache": "/api/cache"
        }
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a query through the multi-layer cache system
    
    The system will:
    1. Check Layer 0 (Exact Cache in Redis)
    2. Check Layer 1 (Semantic Cache in Qdrant)
    3. Check Layer 2 (RAG/Document Cache in Qdrant)
    4. Call LLM if no cache hit
    """
    try:
        result = orchestrator.query(request.query, request.llm_provider)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/api/documents", tags=["Documents"])
async def add_document(request: DocumentRequest):
    """Add a single document to the RAG cache"""
    try:
        doc_id = orchestrator.add_document(request.content, request.metadata)
        return {
            "status": "success",
            "document_id": doc_id,
            "message": "Document added to RAG cache"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding document: {str(e)}"
        )


@app.post("/api/documents/batch", tags=["Documents"])
async def add_documents_batch(request: DocumentBatchRequest):
    """Add multiple documents to the RAG cache in batch"""
    try:
        doc_ids = orchestrator.add_documents_batch(request.documents)
        return {
            "status": "success",
            "document_ids": doc_ids,
            "count": len(doc_ids),
            "message": f"Added {len(doc_ids)} documents to RAG cache"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding documents: {str(e)}"
        )


@app.delete("/api/cache", tags=["Cache Management"])
async def clear_cache(layer: Optional[str] = None):
    """
    Clear cache for specified layer or all layers
    
    - layer: Optional layer number ("0", "1", "2") or None for all layers
    """
    try:
        orchestrator.clear_cache(layer)
        
        if layer:
            message = f"Layer {layer} cache cleared"
        else:
            message = "All cache layers cleared"
        
        return {
            "status": "success",
            "message": message
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health status of all components"""
    try:
        health_status = orchestrator.health_check()
        
        all_healthy = (
            health_status["layer0_redis"] and
            health_status["layer1_qdrant"] and
            health_status["layer2_qdrant"] and
            len(health_status["llm_providers"]) > 0
        )
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            components=health_status
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/api/providers", tags=["LLM"])
async def list_providers():
    """List all available LLM providers"""
    try:
        providers = orchestrator.llm_manager.list_providers()
        return {
            "providers": providers,
            "count": len(providers)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing providers: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

