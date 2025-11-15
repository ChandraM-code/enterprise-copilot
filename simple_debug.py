"""
Simple debug script to test embedding generation
"""

from sentence_transformers import SentenceTransformer
from config import settings

def main():
    print("="*60)
    print("SIMPLE EMBEDDING TEST")
    print("="*60)
    print(f"Model: {settings.embedding_model}")
    print()

    model = SentenceTransformer(settings.embedding_model)
    test_text = "Python uses indentation to define code blocks"

    # Generate embedding twice
    print(f"Test text: '{test_text}'")
    print()

    print("Generating embedding (call 1)...")
    embedding1 = model.encode(test_text)
    print(f"  Type: {type(embedding1)}")
    print(f"  Shape: {embedding1.shape}")
    print(f"  First 5 values: {embedding1[:5]}")
    print()

    print("Generating embedding (call 2)...")
    embedding2 = model.encode(test_text)
    print(f"  Type: {type(embedding2)}")
    print(f"  Shape: {embedding2.shape}")
    print(f"  First 5 values: {embedding2[:5]}")
    print()

    # Check if they're identical
    import numpy as np
    identical = np.array_equal(embedding1, embedding2)
    print(f"Embeddings identical: {identical}")
    print()

    # Now test the actual RAG cache
    print("="*60)
    print("TESTING RAG CACHE")
    print("="*60)

    from cache.layer2_rag_cache import RAGCache

    rag = RAGCache()

    # Clear cache
    print("Clearing RAG cache...")
    rag.clear_all()
    print()

    # Add document
    print(f"Adding document: '{test_text}'")
    doc_id = rag.add_document(test_text, {"test": "debug"})
    print(f"Document ID: {doc_id}")
    print()

    # Query with same text
    print(f"Querying with same text: '{test_text}'")
    results = rag.get(test_text, top_k=1)
    print()

    if results:
        print(f"✓ Found {len(results)} results")
        print(f"  Score: {results[0]['score']}")
        print(f"  Content matches: {results[0]['content'] == test_text}")

        if results[0]['score'] < 0.99:
            print(f"\n❌ ERROR: Score should be ~1.0, got {results[0]['score']}")
        else:
            print(f"\n✓ Score is correct!")
    else:
        print("❌ ERROR: No results found!")

if __name__ == "__main__":
    main()
