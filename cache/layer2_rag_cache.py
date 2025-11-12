from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import time
from config import settings


class RAGCache:
    """
    Layer 2: RAG/Document Cache using Qdrant
    Provides document retrieval and context-based responses
    """
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = "rag_cache"
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize Qdrant collection for RAG cache"""
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
            print(f"Error initializing RAG cache collection: {e}")
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.embedding_model.encode(text).tolist()
    
    def get(self, query: str, top_k: int = 3) -> Optional[List[Dict]]:
        """
        Retrieve relevant documents for the query
        Returns list of relevant documents if similarity exceeds threshold
        """
        query_vector = self._embed_text(query)
        
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=settings.rag_similarity_threshold
            )
            
            if search_results and len(search_results) > 0:
                documents = []
                for result in search_results:
                    documents.append({
                        "content": result.payload.get("content"),
                        "metadata": result.payload.get("metadata", {}),
                        "score": result.score
                    })
                
                print(f"✓ Layer 2 (RAG Cache) HIT - Found {len(documents)} relevant documents")
                return documents
            
            print(f"✗ Layer 2 (RAG Cache) MISS")
            return None
        except Exception as e:
            print(f"Error searching RAG cache: {e}")
            return None
    
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the RAG cache"""
        content_vector = self._embed_text(content)
        doc_id = str(uuid.uuid4())
        
        try:
            point = PointStruct(
                id=doc_id,
                vector=content_vector,
                payload={
                    "content": content,
                    "metadata": metadata or {},
                    "timestamp": time.time()
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print(f"✓ Added document to Layer 2 (RAG Cache) with ID: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"Error adding document to RAG cache: {e}")
            return None
    
    def add_documents_batch(self, documents: List[Dict]) -> List[str]:
        """
        Add multiple documents to RAG cache in batch
        documents: List of dicts with 'content' and optional 'metadata'
        """
        points = []
        doc_ids = []
        
        for doc in documents:
            content = doc.get("content")
            metadata = doc.get("metadata", {})
            
            if not content:
                continue
            
            doc_id = str(uuid.uuid4())
            content_vector = self._embed_text(content)
            
            points.append(PointStruct(
                id=doc_id,
                vector=content_vector,
                payload={
                    "content": content,
                    "metadata": metadata,
                    "timestamp": time.time()
                }
            ))
            doc_ids.append(doc_id)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✓ Added {len(points)} documents to Layer 2 (RAG Cache)")
            return doc_ids
        except Exception as e:
            print(f"Error adding documents to RAG cache: {e}")
            return []
    
    def clear_all(self) -> None:
        """Clear all RAG cache entries"""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            print("✓ Cleared all Layer 2 (RAG Cache) entries")
        except Exception as e:
            print(f"Error clearing RAG cache: {e}")
    
    def health_check(self) -> bool:
        """Check if Qdrant connection is healthy"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False

