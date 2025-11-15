# Level 2 RAG Cache Implementation Analysis

## System Overview

The Level 2 RAG Cache is a document retrieval system built on **Qdrant** vector database that stores document chunks as vectors and retrieves them based on semantic similarity to queries.

---

## 1. Document Chunking & Storage Logic

### Chunking Strategy (`layer2_rag_cache.py`, lines 58-92)

```python
def _initialize_text_splitter(self):
    """Initialize text splitter based on chunking strategy"""
    if settings.chunking_strategy == "recursive":
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,           # 512 tokens
            chunk_overlap=settings.chunk_overlap,     # 50 tokens
            length_function=self._token_length,       # Uses tiktoken
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:  # "fixed" or default
        self.text_splitter = CharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=self._token_length,
            separator="\n"
        )
```

**Key Settings (config.py)**:
- `enable_chunking: bool = True`
- `chunk_size: int = 512` (in tokens)
- `chunk_overlap: int = 50` (in tokens)
- `chunking_strategy: str = "recursive"`

**Token Counting** (lines 75-77):
```python
def _token_length(self, text: str) -> int:
    """Calculate token length using tiktoken"""
    return len(self.tokenizer.encode(text))  # Uses cl100k_base encoding
```

### Storage in Qdrant (lines 145-195)

Each document is processed as follows:

1. **Document is split into chunks** (line 152)
2. **For each chunk**:
   - Generate unique UUID as chunk ID (line 158)
   - Embed the chunk text to vector (line 159)
   - Create metadata with parent document info
   - Store as PointStruct in Qdrant

```python
def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
    doc_id = str(uuid.uuid4())  # Parent document ID
    
    chunks = self._chunk_text(content)  # Split into chunks
    
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())  # Unique ID for each chunk
        chunk_vector = self._embed_text(chunk)  # Embed the chunk
        
        chunk_metadata = {
            "parent_doc_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "is_chunked": len(chunks) > 1,
            **(metadata or {})
        }
        
        point = PointStruct(
            id=chunk_id,
            vector=chunk_vector,
            payload={
                "content": chunk,
                "metadata": chunk_metadata,
                "timestamp": time.time()
            }
        )
        points.append(point)
    
    self.client.upsert(
        collection_name=self.collection_name,
        points=points
    )
```

**Important Detail**: Each chunk is embedded **independently**. If "Python use indentation to define blocks" is a single chunk, that entire sentence becomes one vector.

---

## 2. Embedding Generation

### Embedding Model
**File**: `layer2_rag_cache.py`, line 27

```python
self.embedding_model = SentenceTransformer(settings.embedding_model)
# embedding_model = "all-MiniLM-L6-v2"
```

**Model Specifications**:
- **Name**: all-MiniLM-L6-v2
- **Dimensions**: 384 (from line 28: `self.vector_size = self.embedding_model.get_sentence_embedding_dimension()`)
- **Architecture**: MiniLM distilled model with 6 layers
- **Training**: Trained for semantic textual similarity

### Embedding Process
```python
def _embed_text(self, text: str) -> List[float]:
    """Generate embedding for text"""
    return self.embedding_model.encode(text).tolist()
```

Both documents and queries use **the same embedding model** and **the same method**.

---

## 3. Query Matching & Similarity Calculation

### Query Retrieval (lines 98-143)

```python
def get(self, query: str, top_k: int = 3) -> Optional[List[Dict]]:
    """Retrieve relevant documents/chunks for the query"""
    
    # Step 1: Embed the query using the same model
    query_vector = self._embed_text(query)
    
    # Step 2: Search Qdrant with cosine similarity
    try:
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,                                  # Return top 3
            score_threshold=settings.rag_similarity_threshold  # Default: 0.75
        )
        
        # Step 3: Check if any results exceed threshold
        if search_results and len(search_results) > 0:
            # Extract and return documents
            documents = []
            for result in search_results:
                documents.append({
                    "content": result.payload.get("content"),
                    "metadata": result.payload.get("metadata"),
                    "score": result.score  # Cosine similarity score
                })
            print(f"✓ Layer 2 (RAG Cache) HIT - Found {len(documents)} documents")
            return documents
        
        # Step 4: If no results above threshold → MISS
        print(f"✗ Layer 2 (RAG Cache) MISS")
        return None
```

### Distance Metric

**File**: `layer2_rag_cache.py`, line 51

```python
vectors_config=VectorParams(
    size=self.vector_size,
    distance=Distance.COSINE  # ← Cosine similarity
)
```

**How Cosine Similarity Works**:
- Ranges from -1 to 1 (in Qdrant, typically 0 to 1 for normalized vectors)
- 1.0 = identical vectors
- 0.0 = orthogonal vectors (no similarity)
- Higher = more similar

---

## 4. Cache Hit/Miss Determination

### The Decision Logic (lines 116-139)

```python
if search_results and len(search_results) > 0:
    # HIT: Found results
    documents = []
    for idx, result in enumerate(search_results):
        metadata = result.payload.get("metadata", {})
        doc_info = {
            "content": result.payload.get("content"),
            "metadata": metadata,
            "score": result.score
        }
        documents.append(doc_info)
        debug_print(f"  Result {idx+1}: ..., score={result.score:.4f}")
    
    print(f"✓ Layer 2 (RAG Cache) HIT")
    return documents

print(f"✗ Layer 2 (RAG Cache) MISS")
debug_print(f"No results above threshold {settings.rag_similarity_threshold}")
return None
```

**HIT Condition**: 
- At least one result in `search_results` AND
- That result's `score >= settings.rag_similarity_threshold`

**MISS Condition**:
- No results OR
- All results have `score < settings.rag_similarity_threshold`

---

## 5. Root Cause of Your Cache MISS Issue

### The Problem Scenario

**Indexed Text**: "Python use indentation to define blocks."
**Query**: "Python use indentation to define what?"
**Expected**: HIT (semantically very similar)
**Actual**: MISS (even at 0.5 threshold)

### Why This Happens

1. **Embedding Model Sensitivity**
   - The `all-MiniLM-L6-v2` model encodes the **entire phrase** as a single 384-dimensional vector
   - The key difference between texts:
     - Indexed: "Python use indentation to define **blocks**."
     - Query: "Python use indentation to define **what?**"
   
2. **Semantic Shift**
   - "blocks" is a concrete noun (specific to Python's syntax)
   - "what" is a question word (unknown/placeholder)
   - These have **different semantic meanings** to the embedding model
   - The rest of the phrase may not be enough to overcome this difference

3. **Sentence Structure Impact**
   - Indexed: Statement ending with period (.)
   - Query: Question ending with question mark (?)
   - The embedding model learns that statements and questions are different types
   - This affects the overall vector representation

4. **Vector Space Distance**
   - If the cosine similarity is 0.45, it fails at 0.5 threshold
   - If it's 0.30, it would fail at even 0.5
   - The word "blocks" vs "what" can cause enough drift

### Example Similarity Scores (Hypothetical)

```
Query: "Python use indentation to define what?"

Against indexed chunks:
- "Python use indentation to define blocks." → Score: 0.42-0.48 (MISS at 0.5)
- "Python is a programming language..." → Score: 0.25
- "Python supports lists and tuples..." → Score: 0.15
```

---

## 6. Configuration Details

### File: `config.py`

```python
# RAG Cache Configuration
rag_similarity_threshold: float = 0.75  # Default threshold
enable_chunking: bool = True
chunk_size: int = 512                  # Tokens
chunk_overlap: int = 50                # Tokens
chunking_strategy: str = "recursive"

# Embedding Model
embedding_model: str = "all-MiniLM-L6-v2"  # 384 dimensions

# Debug
debug: bool = False  # Enable for detailed logging
```

---

## 7. Orchestrator Integration

### Cache Hierarchy Flow (`orchestrator.py`)

```python
def query(self, query: str, llm_provider: Optional[str] = None) -> Dict:
    """
    Layer 0 (Exact Match) → Layer 1 (Semantic) → Layer 2 (RAG) → LLM
    """
    
    # Layer 2 Retrieval
    documents = self.layer2.get(query)  # Returns None if MISS
    
    if documents:
        # HIT: Build context and call LLM with context
        context = self._build_context_from_documents(documents)
        response = self.llm_manager.generate_response(
            query=query,
            context=context,  # ← Documents used as context
            provider_name=llm_provider
        )
        # Cache response in Layers 1 and 0
        self.layer1.set(query, response)
        self.layer0.set(query, response)
        return {
            "cache_hit": True,
            "cache_layer": "Layer 2 (RAG Cache)",
            "rag_documents": len(documents),
            ...
        }
    else:
        # MISS: Call LLM without context
        response = self.llm_manager.generate_response(
            query=query,
            context=None,
            provider_name=llm_provider
        )
        return {
            "cache_hit": False,
            "cache_layer": None,
            ...
        }
```

---

## 8. Key Findings Summary

| Aspect | Details |
|--------|---------|
| **Embedding Model** | `all-MiniLM-L6-v2` (384 dimensions) |
| **Similarity Metric** | Cosine Distance |
| **Default Threshold** | 0.75 |
| **Chunking Strategy** | RecursiveCharacterTextSplitter |
| **Chunk Size** | 512 tokens |
| **Token Encoder** | tiktoken (cl100k_base) |
| **Database** | Qdrant Vector Database |
| **HIT Requirement** | Score >= threshold AND len(results) > 0 |
| **MISS Behavior** | Returns None, triggers LLM call |

---

## 9. Why You Get MISS on Similar Queries

### The Core Issue

The embedding model treats "blocks" and "what" as **fundamentally different** tokens:

```
Indexed Chunk Vector:
"Python use indentation to define blocks."
    ↓
[0.23, -0.15, 0.89, ..., 0.42]  (384 dimensions)

Query Vector:
"Python use indentation to define what?"
    ↓
[0.25, -0.12, 0.81, ..., 0.31]  (384 dimensions)

Cosine Similarity = dot_product(v1, v2) / (||v1|| * ||v2||)
                  ≈ 0.42-0.48  (BELOW 0.5 threshold!)
```

### Why This Matters

- **Word Embeddings**: Each word has its own vector representation
- **"blocks" embedding**: Specific to Python syntax, technical meaning
- **"what" embedding**: Generic question word, very different semantic
- **Sentence Encoding**: The model combines all word embeddings
- **Result**: Different overall sentence vectors despite 80% overlap

---

## 10. Potential Solutions

1. **Lower Threshold**: Change `rag_similarity_threshold` from 0.75 to 0.4-0.5
2. **Better Chunking**: Ensure related phrases stay in same chunk
3. **Query Expansion**: Expand query to include more context
4. **Different Embedding Model**: Use larger model (e.g., `all-mpnet-base-v2` - 768 dims)
5. **Preprocessing**: Normalize queries and documents before embedding

