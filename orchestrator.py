from typing import Optional, Dict, List
from cache.layer0_exact_cache import ExactCache
from cache.layer1_semantic_cache import SemanticCache
from cache.layer2_rag_cache import RAGCache
from llm.llm_provider import LLMManager
import time


class CacheOrchestrator:
    """
    Main orchestrator for the multi-layer cache system
    Implements the cache hierarchy: Layer 0 -> Layer 1 -> Layer 2 -> LLM
    """
    
    def __init__(self):
        print("🚀 Initializing Cache Orchestrator...")
        
        # Initialize cache layers
        self.layer0 = ExactCache()
        self.layer1 = SemanticCache()
        self.layer2 = RAGCache()
        
        # Initialize LLM manager
        self.llm_manager = LLMManager()
        
        print("✓ Cache Orchestrator initialized successfully")
    
    def query(self, query: str, llm_provider: Optional[str] = None) -> Dict:
        """
        Process query through the cache hierarchy
        Returns response with metadata about cache hit/miss
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"📝 Processing Query: {query[:100]}...")
        print(f"{'='*60}")
        
        # Layer 0: Check Exact Cache
        print("\n[Layer 0] Checking Exact Cache (Redis)...")
        response = self.layer0.get(query)
        if response:
            elapsed = time.time() - start_time
            return {
                "query": query,
                "response": response,
                "cache_layer": "Layer 0 (Exact Cache)",
                "cache_hit": True,
                "elapsed_time": elapsed,
                "llm_called": False
            }
        
        # Layer 1: Check Semantic Cache
        print("\n[Layer 1] Checking Semantic Cache (Qdrant)...")
        response = self.layer1.get(query)
        if response:
            # Store in Layer 0 for faster future access
            self.layer0.set(query, response)
            
            elapsed = time.time() - start_time
            return {
                "query": query,
                "response": response,
                "cache_layer": "Layer 1 (Semantic Cache)",
                "cache_hit": True,
                "elapsed_time": elapsed,
                "llm_called": False
            }
        
        # Layer 2: Check RAG/Document Cache
        print("\n[Layer 2] Checking RAG/Document Cache (Qdrant)...")
        documents = self.layer2.get(query)
        
        if documents:
            # Build context from retrieved documents
            context = self._build_context_from_documents(documents)
            
            # Generate response with context
            print(f"\n[Layer 2] Found {len(documents)} relevant documents")
            print("[LLM] Generating response with RAG context...")
            
            llm_result = self.llm_manager.generate_response(
                query=query,
                context=context,
                provider_name=llm_provider
            )
            response = llm_result["response"]
            
            # Cache the response in Layer 1 and Layer 0
            self.layer1.set(query, response)
            self.layer0.set(query, response)
            
            elapsed = time.time() - start_time
            return {
                "query": query,
                "response": response,
                "cache_layer": "Layer 2 (RAG Cache)",
                "cache_hit": True,
                "rag_documents": len(documents),
                "elapsed_time": elapsed,
                "llm_called": True,
                "llm_provider": llm_result["provider"]
            }
        
        # No cache hit: Call LLM directly
        print("\n[Cache Miss] No cache hit - calling LLM...")
        llm_result = self.llm_manager.generate_response(
            query=query,
            context=None,
            provider_name=llm_provider
        )
        response = llm_result["response"]
        
        # Cache the response in all layers
        self.layer0.set(query, response)
        self.layer1.set(query, response)
        
        elapsed = time.time() - start_time
        return {
            "query": query,
            "response": response,
            "cache_layer": None,
            "cache_hit": False,
            "elapsed_time": elapsed,
            "llm_called": True,
            "llm_provider": llm_result["provider"]
        }
    
    def _build_context_from_documents(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i} (Relevance: {doc['score']:.2f}):\n{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the RAG cache"""
        return self.layer2.add_document(content, metadata)
    
    def add_documents_batch(self, documents: List[Dict]) -> List[str]:
        """Add multiple documents to RAG cache"""
        return self.layer2.add_documents_batch(documents)
    
    def clear_cache(self, layer: Optional[str] = None) -> None:
        """Clear cache for specified layer or all layers"""
        if layer == "0" or layer is None:
            self.layer0.clear_all()
        if layer == "1" or layer is None:
            self.layer1.clear_all()
        if layer == "2" or layer is None:
            self.layer2.clear_all()
        
        if layer is None:
            print("✓ All cache layers cleared")
    
    def health_check(self) -> Dict:
        """Check health of all components"""
        return {
            "layer0_redis": self.layer0.health_check(),
            "layer1_qdrant": self.layer1.health_check(),
            "layer2_qdrant": self.layer2.health_check(),
            "llm_providers": self.llm_manager.list_providers()
        }

