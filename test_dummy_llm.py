"""
Test function for dummy/custom LLM providers
Uses test_query from test_api.py to test custom LLM implementations
"""

import time
from test_api import test_query, print_section


def test_dummy_llm(llm_provider_name, test_queries=None, verbose=True):
    """
    Test a dummy/custom LLM using the test_query function.

    Args:
        llm_provider_name (str): Name of the custom LLM provider to test
        test_queries (list): List of test queries. If None, uses default queries.
        verbose (bool): Whether to print detailed output

    Returns:
        dict: Test results summary including success rate, response times, and failures
    """

    # Default test queries if none provided
    if test_queries is None:
        test_queries = [
            "What is Python?",
            "Explain machine learning in simple terms",
            "What are the benefits of Docker?",
            "How does caching improve performance?",
            "What is an API?"
        ]

    print_section(f"Testing Dummy LLM: {llm_provider_name}")

    results = {
        "provider": llm_provider_name,
        "total_queries": len(test_queries),
        "successful": 0,
        "failed": 0,
        "response_times": [],
        "failed_queries": [],
        "cache_hits": 0,
        "llm_calls": 0
    }

    for i, query in enumerate(test_queries, 1):
        if verbose:
            print(f"\n[Test {i}/{len(test_queries)}]")
            print("-" * 60)

        start_time = time.time()

        try:
            # Call test_query with the custom provider
            success = test_query(query, llm_provider=llm_provider_name)
            elapsed = time.time() - start_time

            results["response_times"].append(elapsed)

            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["failed_queries"].append({
                    "query": query,
                    "error": "test_query returned False"
                })

        except Exception as e:
            elapsed = time.time() - start_time
            results["failed"] += 1
            results["failed_queries"].append({
                "query": query,
                "error": str(e)
            })
            if verbose:
                print(f"❌ Exception: {e}")

        # Small delay between queries to avoid overwhelming the system
        time.sleep(0.5)

    # Print summary
    print_section("Test Summary")
    print(f"Provider: {llm_provider_name}")
    print(f"Total Queries: {results['total_queries']}")
    print(f"Successful: {results['successful']} ✓")
    print(f"Failed: {results['failed']} ✗")

    if results["response_times"]:
        avg_time = sum(results["response_times"]) / len(results["response_times"])
        min_time = min(results["response_times"])
        max_time = max(results["response_times"])
        print(f"\nResponse Times:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")

    if results["failed_queries"]:
        print(f"\nFailed Queries:")
        for failure in results["failed_queries"]:
            print(f"  - Query: {failure['query']}")
            print(f"    Error: {failure['error']}")

    success_rate = (results['successful'] / results['total_queries']) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    print()

    return results


def test_multiple_providers(provider_names, test_queries=None):
    """
    Test multiple LLM providers and compare results.

    Args:
        provider_names (list): List of provider names to test
        test_queries (list): List of test queries

    Returns:
        dict: Comparison results for all providers
    """

    print_section("Testing Multiple LLM Providers")

    all_results = {}

    for provider in provider_names:
        results = test_dummy_llm(provider, test_queries, verbose=False)
        all_results[provider] = results
        time.sleep(1)

    # Comparison summary
    print_section("Provider Comparison")
    print(f"{'Provider':<20} {'Success Rate':<15} {'Avg Response Time':<20}")
    print("-" * 60)

    for provider, results in all_results.items():
        success_rate = (results['successful'] / results['total_queries']) * 100
        avg_time = sum(results['response_times']) / len(results['response_times']) if results['response_times'] else 0
        print(f"{provider:<20} {success_rate:>6.1f}%         {avg_time:>8.3f}s")

    return all_results


def test_dummy_llm_with_cache_behavior(llm_provider_name, test_query_text):
    """
    Test dummy LLM and observe caching behavior across multiple calls.

    Args:
        llm_provider_name (str): Name of the custom LLM provider
        test_query_text (str): A single query to test multiple times

    Returns:
        dict: Results showing cache behavior
    """

    print_section(f"Testing Cache Behavior for: {llm_provider_name}")

    cache_results = {
        "query": test_query_text,
        "provider": llm_provider_name,
        "iterations": []
    }

    print(f"Query: '{test_query_text}'\n")

    for i in range(3):
        print(f"Iteration {i+1}:")
        print("-" * 40)
        test_query(test_query_text, llm_provider=llm_provider_name)
        cache_results["iterations"].append(i+1)
        print()
        time.sleep(1)

    return cache_results


if __name__ == "__main__":
    """
    Test suite execution for dummy/custom LLM providers
    """

    # Test 1: Test dummy LLM
    print("\n=== Test 1: Test Dummy LLM ===")
    results = test_dummy_llm(
        llm_provider_name="dummy",  # Replace with your dummy LLM name
        test_queries=[
            "What is Python?",
            "Explain APIs",
            "What is Docker?"
        ]
    )

    # Test 2: Test cache behavior
    print("\n=== Test 2: Test Cache Behavior ===")
    cache_results = test_dummy_llm_with_cache_behavior(
        llm_provider_name="dummy",  # Replace with your dummy LLM name
        test_query_text="What is artificial intelligence?"
    )

