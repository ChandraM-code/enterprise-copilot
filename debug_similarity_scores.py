"""
Diagnostic script to analyze similarity scores between queries and documents
This helps identify why RAG cache searches are returning zero results
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import settings
from typing import List, Dict
import sys

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def analyze_similarity_scores():
    """Analyze actual similarity scores between queries and documents"""

    print("=" * 80)
    print("RAG Cache Similarity Score Analysis")
    print("=" * 80)

    # Initialize embedding model
    print(f"\nInitializing embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Initialize Qdrant client
    print(f"\nConnecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_https,
        prefer_grpc=settings.qdrant_prefer_grpc
    )

    # Check if collection exists
    try:
        collection_info = client.get_collection("rag_cache")
        print(f"\n✓ Found RAG cache collection")
        print(f"  - Vectors count: {collection_info.vectors_count}")
        print(f"  - Points count: {collection_info.points_count}")
    except Exception as e:
        print(f"\n✗ Error accessing RAG cache collection: {e}")
        print("  Make sure documents have been added to the cache first")
        return

    print("\n" + "=" * 80)
    print("Current Configuration:")
    print("=" * 80)
    print(f"RAG Similarity Threshold: {settings.rag_similarity_threshold}")
    print(f"Semantic Similarity Threshold: {settings.semantic_similarity_threshold}")
    print(f"Chunking Enabled: {settings.enable_chunking}")
    if settings.enable_chunking:
        print(f"Chunk Size: {settings.chunk_size} tokens")
        print(f"Chunk Overlap: {settings.chunk_overlap} tokens")

    # Test queries of different lengths
    test_queries = [
        "What is Python?",  # Very short
        "Can you explain what Python programming language is?",  # Medium
        "Can you explain what Python programming language is and why it uses indentation?",  # Longer
        "Python was created by Guido van Rossum",  # Direct overlap with doc
    ]

    print("\n" + "=" * 80)
    print("Testing Queries Against RAG Cache")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] \"{query}\"")
        print(f"  Length: {len(query.split())} words")
        print("-" * 80)

        # Generate query embedding
        query_vector = model.encode(query).tolist()

        # Search WITHOUT threshold to see all scores
        try:
            search_results = client.search(
                collection_name="rag_cache",
                query_vector=query_vector,
                limit=5,
                score_threshold=None  # Remove threshold to see all scores
            )

            if search_results:
                print(f"  Found {len(search_results)} results (no threshold applied):")
                for idx, result in enumerate(search_results, 1):
                    content = result.payload.get("content", "")
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    metadata = result.payload.get("metadata", {})

                    print(f"\n  [{idx}] Score: {result.score:.6f}")

                    # Show if this would pass the threshold
                    if result.score >= settings.rag_similarity_threshold:
                        print(f"       Status: ✓ ABOVE threshold ({settings.rag_similarity_threshold})")
                    else:
                        difference = settings.rag_similarity_threshold - result.score
                        print(f"       Status: ✗ BELOW threshold by {difference:.6f}")

                    print(f"       Content: {content_preview}")
                    print(f"       Content length: {len(content.split())} words")

                    if metadata.get("is_chunked"):
                        print(f"       Chunk: {metadata.get('chunk_index', 0)+1}/{metadata.get('total_chunks', 1)}")

                # Now search WITH threshold
                print(f"\n  Searching WITH threshold ({settings.rag_similarity_threshold}):")
                thresholded_results = client.search(
                    collection_name="rag_cache",
                    query_vector=query_vector,
                    limit=5,
                    score_threshold=settings.rag_similarity_threshold
                )

                if thresholded_results:
                    print(f"  ✓ {len(thresholded_results)} results passed threshold")
                else:
                    print(f"  ✗ ZERO results passed threshold")

            else:
                print("  No results found in collection")

        except Exception as e:
            print(f"  Error during search: {e}")

    # Manual similarity calculation test
    print("\n" + "=" * 80)
    print("Manual Similarity Calculation Test")
    print("=" * 80)

    # Create test pairs
    test_text_1 = "What is Python?"
    test_text_2 = "Python is a programming language"
    test_text_3 = "Python was created by Guido van Rossum in 1991. Python's name was inspired by Monty Python."

    emb1 = model.encode(test_text_1)
    emb2 = model.encode(test_text_2)
    emb3 = model.encode(test_text_3)

    sim_1_2 = calculate_cosine_similarity(emb1.tolist(), emb2.tolist())
    sim_1_3 = calculate_cosine_similarity(emb1.tolist(), emb3.tolist())

    print(f"\nText 1 (short query): \"{test_text_1}\"")
    print(f"Text 2 (short doc): \"{test_text_2}\"")
    print(f"Similarity: {sim_1_2:.6f}")
    if sim_1_2 >= settings.rag_similarity_threshold:
        print(f"Status: ✓ ABOVE threshold")
    else:
        print(f"Status: ✗ BELOW threshold by {settings.rag_similarity_threshold - sim_1_2:.6f}")

    print(f"\nText 1 (short query): \"{test_text_1}\"")
    print(f"Text 3 (longer doc): \"{test_text_3}\"")
    print(f"Similarity: {sim_1_3:.6f}")
    if sim_1_3 >= settings.rag_similarity_threshold:
        print(f"Status: ✓ ABOVE threshold")
    else:
        print(f"Status: ✗ BELOW threshold by {settings.rag_similarity_threshold - sim_1_3:.6f}")

    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    print("""
KEY FINDINGS:

1. Cosine similarity scores typically range from 0.0 to 1.0
2. A threshold of 0.75 requires 75% similarity match
3. Short queries vs long documents often have LOWER scores due to:
   - Different semantic density
   - Embedding averaging across different text lengths
   - Limited token overlap doesn't guarantee high cosine similarity

RECOMMENDATIONS:

1. Lower the threshold: Try values between 0.3-0.6 for better recall
2. Use query expansion: Rephrase short queries to be more detailed
3. Adjust chunking: Smaller chunks may better match short queries
4. Consider hybrid search: Combine semantic + keyword matching

To change the threshold, update in config.py or set environment variable:
    RAG_SIMILARITY_THRESHOLD=0.5
""")

if __name__ == "__main__":
    analyze_similarity_scores()
