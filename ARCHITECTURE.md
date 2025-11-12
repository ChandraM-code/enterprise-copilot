# Architecture Documentation

## System Overview

The Agentic Cache-Driven Application is a sophisticated multi-layer caching system designed to optimize response times and reduce LLM API costs through intelligent caching strategies.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                         (main.py)                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cache Orchestrator                           │
│                   (orchestrator.py)                             │
│                                                                 │
│  Manages query flow through cache hierarchy and LLM calls      │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Layer 0       │ │   Layer 1       │ │   Layer 2       │
│  Exact Cache    │ │ Semantic Cache  │ │   RAG Cache     │
│   (Redis)       │ │   (Qdrant)      │ │   (Qdrant)      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼ (if all miss)
                    ┌─────────────────┐
                    │   LLM Manager   │
                    │ (llm_provider)  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              ┌──────────┐      ┌──────────┐
              │  OpenAI  │      │  Gemini  │
              └──────────┘      └──────────┘
```

## Component Details

### 1. FastAPI Server (`main.py`)

**Responsibility**: HTTP API endpoints and request/response handling

**Endpoints**:
- `POST /api/query` - Process queries through cache hierarchy
- `POST /api/documents` - Add single document to RAG cache
- `POST /api/documents/batch` - Add multiple documents
- `DELETE /api/cache` - Clear cache layers
- `GET /api/health` - Health check
- `GET /api/providers` - List available LLM providers

### 2. Cache Orchestrator (`orchestrator.py`)

**Responsibility**: Coordinate query flow through cache layers

**Flow**:
```
Query → Layer 0 → Layer 1 → Layer 2 → LLM
         ↓          ↓          ↓        ↓
       exact    semantic    RAG +    direct
       match     match      LLM      LLM
```

**Key Methods**:
- `query(query, llm_provider)` - Main query processing
- `add_document(content, metadata)` - Add document to RAG
- `clear_cache(layer)` - Clear specific or all layers
- `health_check()` - Check component health

### 3. Layer 0: Exact Cache (`cache/layer0_exact_cache.py`)

**Technology**: Redis

**Purpose**: Ultra-fast exact query matching

**How it works**:
1. Generate MD5 hash of query string
2. Use hash as key: `exact_cache:{hash}`
3. Store response with TTL
4. Return cached response if exists

**Performance**: ~1-5ms

**Key Methods**:
- `get(query)` - Retrieve cached response
- `set(query, response, ttl)` - Store response
- `clear_all()` - Clear all entries

### 4. Layer 1: Semantic Cache (`cache/layer1_semantic_cache.py`)

**Technology**: Qdrant Vector Database

**Purpose**: Find semantically similar queries

**How it works**:
1. Generate embedding vector for query using Sentence Transformers
2. Search Qdrant collection with cosine similarity
3. Return cached response if similarity > threshold (default: 0.85)
4. Store new query-response pairs as vectors

**Performance**: ~10-50ms

**Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)

**Key Methods**:
- `get(query)` - Search for similar queries
- `set(query, response)` - Store query-response pair
- `_embed_query(query)` - Generate embeddings

### 5. Layer 2: RAG Cache (`cache/layer2_rag_cache.py`)

**Technology**: Qdrant Vector Database

**Purpose**: Retrieve relevant documents for context-aware responses

**How it works**:
1. Generate embedding vector for query
2. Search document collection for relevant content
3. Retrieve top-k documents (default: 3) above threshold (0.75)
4. Build context from documents
5. Call LLM with context to generate response
6. Cache response in Layers 0 and 1

**Performance**: ~100-500ms (+ LLM call time)

**Key Methods**:
- `get(query, top_k)` - Retrieve relevant documents
- `add_document(content, metadata)` - Add single document
- `add_documents_batch(documents)` - Add multiple documents

### 6. LLM Manager (`llm/llm_provider.py`)

**Purpose**: Abstract LLM provider interactions

**Supported Providers**:
- OpenAI (GPT-3.5-turbo, GPT-4, etc.)
- Google Gemini (gemini-pro)

**How it works**:
1. Initialize available providers based on API keys
2. Route requests to appropriate provider
3. Handle context injection for RAG queries
4. Return standardized responses

**Key Classes**:
- `BaseLLMProvider` - Abstract base class
- `OpenAIProvider` - OpenAI implementation
- `GeminiProvider` - Gemini implementation
- `LLMManager` - Provider management

## Data Flow

### Scenario 1: First Query (Cache Miss)

```
User Query: "What is machine learning?"
    │
    ▼
Layer 0: ✗ NOT FOUND (exact match)
    │
    ▼
Layer 1: ✗ NOT FOUND (semantic similarity < threshold)
    │
    ▼
Layer 2: ✗ NOT FOUND (no relevant documents)
    │
    ▼
LLM: Generate response
    │
    ▼
Cache in Layer 0, 1 (not 2 - no documents)
    │
    ▼
Return response (Time: ~2000ms)
```

### Scenario 2: Exact Match

```
User Query: "What is machine learning?" (exact same)
    │
    ▼
Layer 0: ✓ FOUND (hash match)
    │
    ▼
Return cached response (Time: ~2ms)
```

### Scenario 3: Similar Query

```
User Query: "Can you explain machine learning?"
    │
    ▼
Layer 0: ✗ NOT FOUND (different text)
    │
    ▼
Layer 1: ✓ FOUND (semantic similarity: 0.92)
    │
    ▼
Cache in Layer 0 for faster future access
    │
    ▼
Return cached response (Time: ~15ms)
```

### Scenario 4: RAG Query

```
User Query: "What is TechCorp's pricing?" (documents exist)
    │
    ▼
Layer 0: ✗ NOT FOUND
    │
    ▼
Layer 1: ✗ NOT FOUND
    │
    ▼
Layer 2: ✓ FOUND (3 relevant documents)
    │
    ▼
Build context from documents
    │
    ▼
LLM: Generate response WITH context
    │
    ▼
Cache in Layer 0, 1
    │
    ▼
Return response (Time: ~1500ms)
```

## Configuration

### Environment Variables

```env
# Redis (Layer 0)
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600

# Qdrant (Layer 1 & 2)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Cache Thresholds
SEMANTIC_SIMILARITY_THRESHOLD=0.85
RAG_SIMILARITY_THRESHOLD=0.75

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Similarity Thresholds

**Semantic Cache (Layer 1)**: 0.85
- Higher values = more strict matching
- Lower values = more cache hits but less accurate
- Recommended range: 0.80 - 0.90

**RAG Cache (Layer 2)**: 0.75
- Higher values = only very relevant documents
- Lower values = more documents, more context
- Recommended range: 0.70 - 0.80

## Performance Characteristics

### Response Time Comparison

| Scenario | Layer | Time | Cost |
|----------|-------|------|------|
| Exact match | Layer 0 | ~2ms | $0 |
| Semantic match | Layer 1 | ~15ms | $0 |
| RAG match | Layer 2 | ~1500ms | ~$0.002 |
| LLM direct | None | ~2000ms | ~$0.002 |

### Cache Hit Rates (Typical)

- Layer 0 (Exact): 30-40% of queries
- Layer 1 (Semantic): 20-30% of remaining
- Layer 2 (RAG): 15-25% of remaining
- LLM Calls: 10-20% of queries

**Result**: ~80-90% of queries served from cache

## Scalability Considerations

### Horizontal Scaling

1. **Redis**: Use Redis Cluster for distributed caching
2. **Qdrant**: Deploy multiple Qdrant nodes
3. **FastAPI**: Deploy multiple API instances behind load balancer

### Vertical Scaling

1. **Redis**: Increase memory allocation
2. **Qdrant**: Increase CPU/memory for faster vector search
3. **Embedding**: Use GPU for faster embedding generation

## Monitoring and Observability

### Key Metrics

1. **Cache Hit Rate**: Percentage of queries served from cache
2. **Response Time**: Average response time per layer
3. **LLM Call Rate**: Number of actual LLM calls
4. **Cost Savings**: Reduction in LLM API costs

### Health Checks

```python
GET /api/health
{
  "status": "healthy",
  "components": {
    "layer0_redis": true,
    "layer1_qdrant": true,
    "layer2_qdrant": true,
    "llm_providers": ["openai", "gemini"]
  }
}
```

## Security Considerations

1. **API Keys**: Store in environment variables, never commit
2. **Rate Limiting**: Implement at FastAPI level
3. **Input Validation**: Sanitize all user inputs
4. **Authentication**: Add API key/JWT authentication for production
5. **HTTPS**: Use TLS for all communications

## Future Enhancements

1. **Layer 0.5**: Add in-memory LRU cache for ultra-fast access
2. **Streaming**: Support streaming responses from LLMs
3. **Analytics**: Add detailed analytics dashboard
4. **Multi-tenancy**: Support multiple organizations/users
5. **Cache Warming**: Pre-populate cache with common queries
6. **A/B Testing**: Test different LLM providers/models
7. **Feedback Loop**: Learn from user feedback to improve caching

## Troubleshooting

### Issue: High LLM Call Rate

**Possible Causes**:
- Thresholds too high
- Insufficient documents in RAG cache
- Queries too diverse

**Solutions**:
- Lower semantic/RAG thresholds
- Add more documents to RAG cache
- Analyze query patterns

### Issue: High Cache Miss Rate

**Possible Causes**:
- Low TTL
- Cache cleared frequently
- New query patterns

**Solutions**:
- Increase TTL
- Review cache clearing policies
- Add common queries manually

### Issue: Slow Response Times

**Possible Causes**:
- Qdrant overloaded
- Large embedding model
- Network latency

**Solutions**:
- Scale Qdrant resources
- Use faster embedding model
- Deploy closer to users

## Conclusion

This architecture provides a robust, scalable solution for reducing LLM costs and improving response times through intelligent multi-layer caching. The hierarchical approach ensures optimal performance while maintaining accuracy and relevance.

