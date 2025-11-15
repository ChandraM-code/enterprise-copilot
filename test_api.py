"""
Test script for the Agentic Cache-Driven Application
Run this after starting the application to test all endpoints
"""

import requests
import time
import json


BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_health_check():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_list_providers():
    """Test list providers endpoint"""
    print_section("Testing List Providers")
    
    response = requests.get(f"{BASE_URL}/api/providers")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_add_documents():
    """Test adding documents to RAG cache"""
    print_section("Testing Add Documents")
    
    documents = [
        {
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "metadata": {"topic": "AI", "subtopic": "machine-learning"}
        },
        {
            "content": "Docker is a platform that uses containerization to package applications and their dependencies together, ensuring consistency across different environments.",
            "metadata": {"topic": "DevOps", "tool": "docker"}
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/documents/batch",
        json={"documents": documents}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_query(query, llm_provider=None):
    """Test query endpoint"""
    payload = {"query": query}
    if llm_provider:
        payload["llm_provider"] = llm_provider
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/query",
        json=payload
    )
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"Query: {query}")
        print(f"Cache Layer: {result.get('cache_layer', 'None')}")
        print(f"Cache Hit: {result.get('cache_hit')}")
        print(f"LLM Called: {result.get('llm_called')}")
        if result.get('llm_provider'):
            print(f"LLM Provider: {result.get('llm_provider')}")
        print(f"Response Time: {elapsed:.3f}s")
        print(f"Response: {result.get('response')[:200]}...")
        print()
        return True
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def test_cache_hierarchy():
    """Test the cache hierarchy with multiple queries"""
    print_section("Testing Cache Hierarchy")
    
    # First query - should call LLM
    print("1. First query (Expected: Cache MISS, LLM call):")
    test_query("What is Python programming language?")
    time.sleep(1)
    
    # Same query - should hit Layer 0 (Exact Cache)
    print("\n2. Exact same query (Expected: Layer 0 HIT):")
    test_query("What is Python programming language?")
    time.sleep(1)
    
    # Similar query - should hit Layer 1 (Semantic Cache)
    print("\n3. Similar query (Expected: Layer 1 HIT):")
    test_query("Can you explain the Python programming language?")
    time.sleep(1)
    
    # Query about pre-loaded document - should hit Layer 2 (RAG Cache)
    print("\n4. Query about pre-loaded document (Expected: Layer 2 HIT):")
    test_query("Tell me about machine learning")
    time.sleep(1)


def test_clear_cache():
    """Test cache clearing"""
    print_section("Clearing Cache")
    
    # Clear Layer 0
    response = requests.delete(f"{BASE_URL}/api/cache?layer=0")
    print(f"Clear Layer 0 - Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Clear all layers
    response = requests.delete(f"{BASE_URL}/api/cache")
    print(f"\nClear All Layers - Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  AGENTIC CACHE-DRIVEN APPLICATION - TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Health Check
        if not test_health_check():
            print("❌ Health check failed. Make sure the application is running.")
            return
        
        time.sleep(1)
        
        # Test 2: List Providers
        if not test_list_providers():
            print("⚠️  Warning: Could not list providers. Check API keys.")
        
        time.sleep(1)
        
        # Test 3: Add Documents
        print("\nAdding sample documents to RAG cache...")
        test_add_documents()
        
        time.sleep(2)
        
        # Test 4: Test Cache Hierarchy
        test_cache_hierarchy()
        
        time.sleep(1)
        
        # Test 5: Clear Cache
        test_clear_cache()
        
        print("\n" + "="*60)
        print("  ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API.")
        print("Make sure the application is running at http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

