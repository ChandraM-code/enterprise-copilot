from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import uuid
import time
from config import settings

"""
    Layer 1: Semantic Cache using Qdrant
    Provides semantic similarity search for similar queries
"""
class SemanticCache:
    
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = "semantic_cache"
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._initialize_collection()

    """Initialize Qdrant collection for semantic cache"""
    def _initialize_collection(self):
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing semantic cache collection: {e}")
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        return self.embedding_model.encode(query).tolist()
    
    """
        Retrieve cached response for semantically similar query
        Returns response if similarity score exceeds threshold
    """
    def get(self, query: str) -> Optional[str]:
        
        query_vector = self._embed_query(query)
        
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=1,
                score_threshold=settings.semantic_similarity_threshold
            )
            
            if search_result and len(search_result) > 0:
                result = search_result[0]
                print(f"✓ Layer 1 (Semantic Cache) HIT with score: {result.score:.4f}")
                return result.payload.get("response")
            
            print(f"✗ Layer 1 (Semantic Cache) MISS")
            return None
        except Exception as e:
            print(f"Error searching semantic cache: {e}")
            return None
    
    def set(self, query: str, response: str) -> None:
        """Store query-response pair in semantic cache"""
        query_vector = self._embed_query(query)
        
        try:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vector,
                payload={
                    "query": query,
                    "response": response,
                    "timestamp": time.time()
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print(f"✓ Stored in Layer 1 (Semantic Cache)")
        except Exception as e:
            print(f"Error storing in semantic cache: {e}")
    
    def clear_all(self) -> None:
        """Clear all semantic cache entries"""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            print("✓ Cleared all Layer 1 (Semantic Cache) entries")
        except Exception as e:
            print(f"Error clearing semantic cache: {e}")
    
    def health_check(self) -> bool:
        """Check if Qdrant connection is healthy"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False

