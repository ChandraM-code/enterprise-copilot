"""
Test function for dummy/custom LLM providers
Uses test_query from test_api.py to test custom LLM implementations
"""

import time
import json
import requests
from test_api import test_query, print_section, test_clear_cache

BASE_URL = "http://localhost:8000"

def add_documents():
    """Adding documents to RAG cache"""
    print_section("Add Documents")
    
    documents = [
        {
            "content": "Python was created by Guido van Rossum in 1991. Python’s name was inspired by Monty Python. Python uses indentation to define code blocks. Python supports multiple programming paradigms. Python’s package manager is called pip. Python lists are mutable. Python tuples are immutable. Python dictionaries store key-value pairs. Python uses garbage collection for memory management. Python integers have arbitrary precision. Python strings are immutable. Python functions are first-class objects. Python supports lambda expressions. Python’s standard file extension is .py. Python uses dynamic typing. Python includes a built-in REPL. Python supports list comprehensions. Python exceptions propagate up the call stack. Python generators use the yield keyword. Python decorators wrap functions for modification. Python supports multiple inheritance. Python’s standard library includes datetime. Python’s collections module offers specialized data types. Python uses the Global Interpreter Lock in CPython. Python’s virtual environments isolate dependencies. Python can interface with C using Cython. Python’s asyncio supports asynchronous programming. Python booleans are subclasses of integers. Python classes can define custom magic methods. Python uses duck typing. Python’s with statement manages context. Python’s json module handles JSON data. Python supports binary literals using 0b. Python supports f-strings for interpolation. Python’s itertools module provides iterator tools. Python uses hash tables for dictionary storage. Python modules can be grouped into packages. Python’s default encoding is UTF-8. Python supports type hints using typing. Python’s os module provides OS-level functions. Python supports set comprehensions. Python integers are immutable. Python allows unpacking of iterables. Python’s math module provides mathematical functions. Python objects have reference counts. Python list slicing creates a new list. Python’s zip function aggregates iterables. Python’s sum function performs numeric addition. Python supports generator expressions. Python’s subprocess module runs external commands.",
            "metadata": {"topic": "programming", "language": "Python"}
        },
        {
            "content": "India’s capital is New Delhi. Japan’s capital is Tokyo. France’s capital is Paris. Germany’s capital is Berlin. Australia’s capital is Canberra. Canada’s capital is Ottawa. Brazil’s capital is Brasília. China’s capital is Beijing. Russia’s capital is Moscow. Italy’s capital is Rome. Spain’s capital is Madrid. South Korea’s capital is Seoul. United Kingdom’s capital is London. United States’ capital is Washington, D.C. Mexico’s capital is Mexico City. Argentina’s capital is Buenos Aires. Egypt’s capital is Cairo. South Africa’s capital is Pretoria. Kenya’s capital is Nairobi. Nigeria’s capital is Abuja. Saudi Arabia’s capital is Riyadh. Turkey’s capital is Ankara. Thailand’s capital is Bangkok. Indonesia’s capital is Jakarta. Vietnam’s capital is Hanoi. Philippines’ capital is Manila. Malaysia’s capital is Kuala Lumpur. Singapore’s capital is Singapore. New Zealand’s capital is Wellington. Pakistan’s capital is Islamabad. Sri Lanka’s capital is Colombo. Bangladesh’s capital is Dhaka. Nepal’s capital is Kathmandu. Bhutan’s capital is Thimphu. Norway’s capital is Oslo. Sweden’s capital is Stockholm. Finland’s capital is Helsinki. Denmark’s capital is Copenhagen. Netherlands’ capital is Amsterdam. Belgium’s capital is Brussels. Switzerland’s capital is Bern. Austria’s capital is Vienna. Poland’s capital is Warsaw. Czech Republic’s capital is Prague. Greece’s capital is Athens. Portugal’s capital is Lisbon. Ireland’s capital is Dublin. Chile’s capital is Santiago. Peru’s capital is Lima. Colombia’s capital is Bogotá.",
            "metadata": {"topic": "trivia", "category": "Capitals"}
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/documents/batch",
        json={"documents": documents}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

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

    # Step 1: Add documents to RAG cache
    add_documents()
    time.sleep(2)  # Wait for documents to be indexed

    test_clear_cache()
    time.sleep(2)  # Wait for cache to clear
    
    # Test 1: Test dummy LLM
    print("\n=== Test 1: Test Dummy LLM ===")
    results = test_dummy_llm(
        llm_provider_name="dummy",  # my dummy LLM name
        test_queries=[
            "Why does python use indentation?",
            "What is python's package manager called?",
            "What is France's capital?"
        ]
    )

    # Test 2: Test cache behavior
    print("\n=== Test 2: Test Cache Behavior: Repeated Query ===")
    cache_results = test_dummy_llm_with_cache_behavior(
        llm_provider_name="dummy",  # my dummy LLM name
        test_query_text="What is a key characteristic of Python Lists?"
    )

    print("\n=== Test 3: Test Cache Behavior: Semantically Similar Query ===")
    cache_results = test_dummy_llm_with_cache_behavior(
        llm_provider_name="dummy",  # my dummy LLM name
        test_query_text="Can you tell me a key characteristic of Python Lists?"
    )

    test_clear_cache()
    print("\nCache tests completed.")


