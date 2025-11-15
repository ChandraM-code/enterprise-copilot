"""
Debug script to diagnose embedding and similarity issues in RAG cache
Tests if embeddings are generated consistently and similarity calculations work correctly
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import settings
import uuid


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def test_embedding_consistency():
    """Test if same text produces same embedding"""
    print("="*60)
    print("TEST 1: Embedding Consistency")
    print("="*60)

    model = SentenceTransformer(settings.embedding_model)
    test_text = "Python uses indentation to define code blocks"

    # Generate embedding twice
    embedding1 = model.encode(test_text)
    embedding2 = model.encode(test_text)

    # Check if identical
    are_identical = np.array_equal(embedding1, embedding2)
    similarity = cosine_similarity(embedding1, embedding2)

    print(f"Test text: '{test_text}'")
    print(f"Embedding dimension: {len(embedding1)}")
    print(f"Embeddings identical: {are_identical}")
    print(f"Cosine similarity: {similarity:.10f}")
    print(f"First 5 values of embedding1: {embedding1[:5]}")
    print(f"First 5 values of embedding2: {embedding2[:5]}")

    if similarity < 0.99:
        print("❌ ERROR: Same text produces different embeddings!")
        return False
    else:
        print("✓ Same text produces consistent embeddings")
        return True


def test_encode_parameters():
    """Test different encode() parameters"""
    print("\n" + "="*60)
    print("TEST 2: Encode Parameters")
    print("="*60)

    model = SentenceTransformer(settings.embedding_model)
    test_text = "Python uses indentation to define code blocks"

    # Test with default parameters
    emb_default = model.encode(test_text)
    print(f"\nDefault encode():")
    print(f"  Shape: {emb_default.shape}")
    print(f"  Dtype: {emb_default.dtype}")
    print(f"  First 5 values: {emb_default[:5]}")
    print(f"  L2 norm: {np.linalg.norm(emb_default):.6f}")

    # Test with normalize_embeddings=True
    emb_normalized = model.encode(test_text, normalize_embeddings=True)
    print(f"\nWith normalize_embeddings=True:")
    print(f"  Shape: {emb_normalized.shape}")
    print(f"  Dtype: {emb_normalized.dtype}")
    print(f"  First 5 values: {emb_normalized[:5]}")
    print(f"  L2 norm: {np.linalg.norm(emb_normalized):.6f}")

    # Compare
    similarity = cosine_similarity(emb_default, emb_normalized)
    print(f"\nCosine similarity between default and normalized: {similarity:.10f}")

    return emb_default, emb_normalized


def test_qdrant_search_with_same_vector():
    """Test if Qdrant returns similarity of 1.0 when searching with same vector"""
    print("\n" + "="*60)
    print("TEST 3: Qdrant Same Vector Search")
    print("="*60)

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_https,
        prefer_grpc=settings.qdrant_prefer_grpc
    )

    model = SentenceTransformer(settings.embedding_model)
    test_collection = "debug_test_collection"

    # Clean up if exists
    try:
        client.delete_collection(test_collection)
    except:
        pass

    # Create test collection
    vector_size = model.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"✓ Created test collection with vector size: {vector_size}")

    # Add a document
    test_text = "Python uses indentation to define code blocks"
    doc_embedding = model.encode(test_text).tolist()

    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=test_collection,
        points=[
            PointStruct(
                id=point_id,
                vector=doc_embedding,
                payload={"content": test_text}
            )
        ]
    )
    print(f"✓ Added document with ID: {point_id}")

    # Search with exact same embedding
    query_embedding = model.encode(test_text).tolist()
    results = client.search(
        collection_name=test_collection,
        query_vector=query_embedding,
        limit=1
    )

    print(f"\nSearch with identical text:")
    print(f"  Document text: '{test_text}'")
    print(f"  Query text: '{test_text}'")
    if results:
        print(f"  Similarity score: {results[0].score:.10f}")
        print(f"  Expected: ~1.0")
        if results[0].score < 0.99:
            print(f"  ❌ ERROR: Similarity should be ~1.0, got {results[0].score}")
        else:
            print(f"  ✓ Similarity is correct")
    else:
        print(f"  ❌ ERROR: No results returned!")

    # Clean up
    client.delete_collection(test_collection)

    return results[0].score if results else None


def test_rag_cache_encoding():
    """Test the actual RAG cache encoding methods"""
    print("\n" + "="*60)
    print("TEST 4: RAG Cache Encoding Methods")
    print("="*60)

    from cache.layer2_rag_cache import RAGCache

    rag_cache = RAGCache()
    test_text = "Python uses indentation to define code blocks"

    # Test _embed_text method
    embedding1 = rag_cache._embed_text(test_text)
    embedding2 = rag_cache._embed_text(test_text)

    print(f"Test text: '{test_text}'")
    print(f"Embedding dimension: {len(embedding1)}")
    print(f"First 5 values (call 1): {embedding1[:5]}")
    print(f"First 5 values (call 2): {embedding2[:5]}")

    # Calculate similarity
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Cosine similarity: {similarity:.10f}")

    if similarity < 0.99:
        print("❌ ERROR: RAG cache encoding is inconsistent!")
        return False
    else:
        print("✓ RAG cache encoding is consistent")
        return True


def test_full_workflow():
    """Test the full add document -> query workflow"""
    print("\n" + "="*60)
    print("TEST 5: Full Workflow (Add Document -> Query)")
    print("="*60)

    from cache.layer2_rag_cache import RAGCache

    rag_cache = RAGCache()

    # Clear cache first
    print("Clearing RAG cache...")
    rag_cache.clear_all()

    # Add a document
    test_text = "Python uses indentation to define code blocks"
    print(f"\nAdding document: '{test_text}'")
    doc_id = rag_cache.add_document(test_text, {"test": "debug"})
    print(f"Document ID: {doc_id}")

    # Query with exact same text
    print(f"\nQuerying with exact same text: '{test_text}'")
    results = rag_cache.get(test_text, top_k=1)

    if results:
        print(f"✓ Found {len(results)} results")
        print(f"  Top result similarity: {results[0]['score']:.10f}")
        print(f"  Content: '{results[0]['content']}'")
        print(f"  Content matches: {results[0]['content'] == test_text}")

        if results[0]['score'] < 0.99:
            print(f"  ❌ ERROR: Similarity should be ~1.0 for identical text, got {results[0]['score']}")
            return False
        else:
            print(f"  ✓ Similarity is correct")
            return True
    else:
        print(f"❌ ERROR: No results found for identical text!")
        return False


def test_with_chunking():
    """Test with chunking enabled/disabled"""
    print("\n" + "="*60)
    print("TEST 6: Chunking Impact")
    print("="*60)

    from cache.layer2_rag_cache import RAGCache

    # Save original setting
    original_chunking = settings.enable_chunking

    for chunking_enabled in [False, True]:
        settings.enable_chunking = chunking_enabled
        print(f"\n--- Chunking: {'ENABLED' if chunking_enabled else 'DISABLED'} ---")

        rag_cache = RAGCache()
        rag_cache.clear_all()

        test_text = "Python uses indentation to define code blocks"
        print(f"Adding: '{test_text}'")
        doc_id = rag_cache.add_document(test_text, {"test": "chunking"})

        print(f"Querying: '{test_text}'")
        results = rag_cache.get(test_text, top_k=1)

        if results:
            print(f"  Score: {results[0]['score']:.10f}")
        else:
            print(f"  ❌ No results!")

    # Restore original setting
    settings.enable_chunking = original_chunking


def main():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("EMBEDDING DIAGNOSTIC TESTS")
    print("="*60)
    print(f"Model: {settings.embedding_model}")
    print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print("="*60)

    test_embedding_consistency()
    test_encode_parameters()
    test_qdrant_search_with_same_vector()
    test_rag_cache_encoding()
    test_full_workflow()
    test_with_chunking()

    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
