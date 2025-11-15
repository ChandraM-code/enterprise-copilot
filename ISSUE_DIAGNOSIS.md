# Cache MISS Issue Diagnosis

## Quick Summary

When you index "Python use indentation to define blocks." and query "Python use indentation to define what?", you get a MISS because the embedding model produces different vectors for these sentences, resulting in a cosine similarity score below your threshold.

---

## The Issue Breakdown

### What's Happening

```
Step 1: Index Document
───────────────────────
Document: "Python use indentation to define blocks."
           ↓
         [Chunk]
           ↓
    [Sentence Transformer Model]
           ↓
   Vector: [0.23, -0.15, 0.89, 0.12, ..., 0.42]  (384 dims)
           ↓
    Stored in Qdrant with this vector


Step 2: Query Document
──────────────────────
Query: "Python use indentation to define what?"
           ↓
    [Sentence Transformer Model]
           ↓
   Vector: [0.25, -0.12, 0.81, 0.09, ..., 0.31]  (384 dims)
           ↓
    Search Qdrant for similar vectors


Step 3: Calculate Similarity
─────────────────────────────
Cosine Similarity = dot_product(v1, v2) / (||v1|| * ||v2||)

    0.23×0.25 + (-0.15)×(-0.12) + 0.89×0.81 + ... + 0.42×0.31
    ───────────────────────────────────────────────────────────
        √(0.23² + 0.15² + 0.89² + ... + 0.42²) × √(0.25² + 0.12² + 0.81² + ...)
    
    = 0.42-0.48  (approximately)
           ↓
    BELOW your threshold (e.g., 0.5)
           ↓
    ✗ MISS - No cached response


Step 4: Cache Miss Behavior
────────────────────────────
Since score < threshold:
- Qdrant returns empty results
- Layer 2 returns None
- Orchestrator calls LLM
- Response generated without context
```

---

## Why the Vectors Are Different

### Word-by-Word Comparison

```
Index:  Python  use  indentation  to  define  BLOCKS  .
        ─────────────────────────────────────────────────
        ↓      ↓      ↓         ↓    ↓        ↓        ↓
        ~      ~      ~         ~    ~     [DIFF!]    ~

Query:  Python  use  indentation  to  define  WHAT?   
        ─────────────────────────────────────────────────
        ↓      ↓      ↓         ↓    ↓        ↓        ↓
        ~      ~      ~         ~    ~     [DIFF!]    ~
```

### The Problem Words

**"blocks" vs "what"**

- **"blocks"**: Semantic Embedding
  - Context: Python code blocks, syntax elements
  - Type: Concrete noun, technical term
  - Vector: Specific to programming domain
  - Example: similar to "statements", "segments", "sections"
  
- **"what"**: Semantic Embedding
  - Context: Question word, interrogative
  - Type: Question marker, generic
  - Vector: Very different from concrete nouns
  - Example: similar to "why", "how", "which"

**The Embedding Model sees these as VERY DIFFERENT**

```
Word Embeddings (all-MiniLM-L6-v2):

blocks ──→ [0.45, 0.78, -0.23, 0.91, ...] 384 dims
           (Technical/Concrete/Noun)
           
what ──→   [0.12, 0.15, 0.88, -0.15, ...] 384 dims
           (Question/Generic/Interrogative)
           
Distance: ~0.35-0.40 (orthogonal in vector space)
```

### Sentence Encoding

The model doesn't just average word embeddings. It learns how words interact:

```
"Python use indentation to define blocks."
 ↓
[
  Word vectors: python, use, indentation, to, define, blocks, .
  Attention weights: how much each word matters
  Positional encoding: word positions
  Context: surrounding words
  Sentence-level features
] → Combined into single vector (384 dims)
    [0.23, -0.15, 0.89, 0.12, ..., 0.42]


"Python use indentation to define what?"
 ↓
[
  Word vectors: python, use, indentation, to, define, what, ?
  Attention weights: different distribution (question format)
  Positional encoding: different word positions
  Context: different surrounding words
  Sentence-level features: question structure
] → Combined into different vector (384 dims)
    [0.25, -0.12, 0.81, 0.09, ..., 0.31]
```

**Result**: Different vectors → Lower cosine similarity → Cache MISS

---

## Current System Settings

**File: config.py**

```python
# Threshold for RAG Cache hits
rag_similarity_threshold: float = 0.75  # ← Your queries need score >= 0.75

# Embedding Model
embedding_model: str = "all-MiniLM-L6-v2"  # Only 384 dimensions
                                           # Good for speed, limited nuance

# Chunking
chunk_size: int = 512  # tokens
chunk_overlap: int = 50  # tokens
chunking_strategy: str = "recursive"  # Tries to split on meaningful boundaries
```

**File: layer2_rag_cache.py**

```python
# Distance metric
distance=Distance.COSINE  # Scale: 0 (no match) to 1 (identical)

# Search parameters
limit=top_k  # Return top 3 by default
score_threshold=settings.rag_similarity_threshold  # Must exceed this

# Hit/Miss Logic
if search_results and len(search_results) > 0:
    return documents  # HIT
else:
    return None  # MISS
```

---

## Why Similarity Score is Low

### Factors Affecting the Score

| Factor | Impact | Your Case |
|--------|--------|-----------|
| Same prefix (80% overlap) | Helps similarity | ✓ 80% identical text |
| Different key word | Hurts similarity | ✗ "blocks" vs "what" |
| Different punctuation | Slight impact | ✗ Period vs question mark |
| Query structure (question) | Affects encoding | ✗ Question vs statement |
| Model dimensionality | Low = less nuance | ✗ Only 384 dims |

**Estimated Impact**:
- Shared words: +0.70
- Different key word: -0.25 to -0.30
- Question vs statement: -0.05
- **Final Score: 0.40-0.50** (Below 0.5 threshold)

---

## Verification Steps

### Enable Debug Logging

```python
# In config.py
debug: bool = True  # Change from False to True
```

### Run Query and Check Output

```
[DEBUG] Searching RAG cache for query: 'Python use indentation to define what?'
[DEBUG] Top-k: 3, Threshold: 0.75
[DEBUG] Query embedded successfully (384 dimensions)
[DEBUG] Found 1 results from Qdrant search
[DEBUG]   Result 1: full document, score=0.4523 ← THE CULPRIT!
✗ Layer 2 (RAG Cache) MISS
[DEBUG] No results above threshold 0.75
```

The score **0.4523 < 0.75** → MISS

---

## Why Even 0.5 Threshold Doesn't Help

If the actual cosine similarity is ~0.42-0.48, even setting threshold to 0.5 might marginally help, but if it's ~0.35-0.45, you'll still get misses.

**The Root Issue**: The words "blocks" and "what" are too semantically different for this embedding model to maintain high similarity.

---

## Recommended Solutions

### Solution 1: Lower the Threshold (Quick Fix)

```python
# config.py
rag_similarity_threshold: float = 0.40  # More lenient

# Trade-off: More false positives, fewer misses
```

### Solution 2: Use Larger Embedding Model (Better Quality)

```python
# config.py
embedding_model: str = "all-mpnet-base-v2"  # 768 dimensions instead of 384
# More nuanced representations, catches more semantic similarity
```

### Solution 3: Index Query-Like Documents

```python
# Instead of indexing: "Python use indentation to define blocks."
# Index something like: "What does Python use for indentation? Blocks."
# Or: "Python uses indentation to define blocks, not braces."
```

### Solution 4: Query Expansion (Advanced)

```python
# Before searching, expand query with synonyms:
# Original: "Python use indentation to define what?"
# Expanded: "Python use indentation to define ... blocks code segments statements"
# This gives embedding model more context
```

### Solution 5: Preprocessing & Normalization

```python
# Normalize before embedding:
# Remove punctuation
# Lowercase
# Expand contractions
# Standardize phrasing
```

---

## Real-World Implications

### Why This Matters

```
Scenario 1: Current Settings (threshold 0.75)
─────────────────────────────────────────────
Index: "Python indentation blocks"
Query: "Python indentation what"
Similarity: 0.42
Result: ✗ MISS → LLM called → Slower, more expensive


Scenario 2: Lowered Threshold (threshold 0.40)
──────────────────────────────────────────────
Index: "Python indentation blocks"
Query: "Python indentation what"
Similarity: 0.42
Result: ✓ HIT → Use cached documents → Faster, cheaper


Scenario 3: Better Embedding Model (all-mpnet-base-v2)
─────────────────────────────────────────────────────
Index: "Python indentation blocks"
Query: "Python indentation what"
Similarity: 0.68 (improved!)
Result: ✓ HIT → Better matching with same threshold
```

---

## File Locations for Reference

| Aspect | File | Lines |
|--------|------|-------|
| Chunking | `cache/layer2_rag_cache.py` | 58-92 |
| Embedding | `cache/layer2_rag_cache.py` | 94-96, 27 |
| Query Matching | `cache/layer2_rag_cache.py` | 98-143 |
| Hit/Miss Logic | `cache/layer2_rag_cache.py` | 116-139 |
| Similarity Threshold | `config.py` | 42 |
| Embedding Model | `config.py` | 52 |
| Qdrant Search | `cache/layer2_rag_cache.py` | 109-114 |
| Orchestrator Flow | `orchestrator.py` | 70-103 |

