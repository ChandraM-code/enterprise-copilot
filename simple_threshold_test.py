"""
Simple test to check what's happening with the RAG cache threshold
This connects directly to your running Qdrant instance
"""

from qdrant_client import QdrantClient
from config import settings

def check_threshold_issue():
    print("=" * 80)
    print("RAG Cache Threshold Issue Diagnosis")
    print("=" * 80)

    print(f"\nCurrent RAG Similarity Threshold: {settings.rag_similarity_threshold}")
    print(f"Current Semantic Similarity Threshold: {settings.semantic_similarity_threshold}")

    # Connect to Qdrant
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_https,
            prefer_grpc=settings.qdrant_prefer_grpc
        )

        print(f"\n✓ Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")

        # Check collections
        collections = client.get_collections().collections
        print(f"\nAvailable collections:")
        for col in collections:
            print(f"  - {col.name}")

        # Check RAG cache collection
        try:
            rag_info = client.get_collection("rag_cache")
            print(f"\nRAG Cache Collection Info:")
            print(f"  - Vector count: {rag_info.vectors_count}")
            print(f"  - Points count: {rag_info.points_count}")
            print(f"  - Distance metric: {rag_info.config.params.vectors.distance}")

            if rag_info.points_count == 0:
                print("\n⚠️  WARNING: RAG cache is empty! Add documents first.")
                return

            # Get a sample point to understand the data
            sample = client.scroll(
                collection_name="rag_cache",
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if sample[0]:
                print(f"\nSample document in cache:")
                payload = sample[0][0].payload
                content = payload.get("content", "")
                print(f"  - Content preview: {content[:150]}...")
                print(f"  - Content length: {len(content.split())} words")
                print(f"  - Metadata: {payload.get('metadata', {})}")

        except Exception as e:
            print(f"\n✗ Error accessing RAG cache: {e}")

    except Exception as e:
        print(f"\n✗ Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running on the configured host and port")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print(f"""
The issue is likely that your threshold of {settings.rag_similarity_threshold} is TOO HIGH
for matching short queries (10-15 tokens) against long documents (250-300 tokens).

With cosine similarity:
- Identical text pairs: ~0.95-1.00
- Very similar same-length text: ~0.80-0.95
- Related same-length text: ~0.60-0.80
- Short query vs long document (even with overlap): ~0.30-0.60 ⚠️

RECOMMENDED FIXES:

1. IMMEDIATE FIX - Lower the threshold in config.py:
   rag_similarity_threshold: float = 0.50  # or even 0.40

2. BETTER FIX - Adjust chunking for shorter chunks:
   chunk_size: int = 100  # matches query length better

3. TEST FIX - Temporarily set threshold to 0.0 to see all scores:
   This will show you what scores you're actually getting

After making changes, restart your application and test again.
""")

if __name__ == "__main__":
    check_threshold_issue()
