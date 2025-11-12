from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import tiktoken
import uuid
import time
from config import settings, debug_print


class RAGCache:
    """
    Layer 2: RAG/Document Cache using Qdrant
    Provides document retrieval and context-based responses
    """
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_https,
            prefer_grpc=settings.qdrant_prefer_grpc
        )
        self.collection_name = "rag_cache"
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize text splitter for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._initialize_text_splitter()

        debug_print(f"RAG Cache initialized with chunking={'enabled' if settings.enable_chunking else 'disabled'}")
        debug_print(f"Chunk size: {settings.chunk_size}, Overlap: {settings.chunk_overlap}, Strategy: {settings.chunking_strategy}")

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
    
    def _initialize_text_splitter(self):
        """Initialize text splitter based on chunking strategy"""
        if settings.chunking_strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=self._token_length,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:  # "fixed" or default
            self.text_splitter = CharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=self._token_length,
                separator="\n"
            )

    def _token_length(self, text: str) -> int:
        """Calculate token length using tiktoken"""
        return len(self.tokenizer.encode(text))

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on configuration"""
        if not settings.enable_chunking:
            debug_print(f"Chunking disabled, returning full text (length: {len(text)} chars)")
            return [text]

        chunks = self.text_splitter.split_text(text)
        debug_print(f"Split text into {len(chunks)} chunks using {settings.chunking_strategy} strategy")

        for i, chunk in enumerate(chunks):
            token_count = self._token_length(chunk)
            debug_print(f"  Chunk {i+1}: {token_count} tokens, {len(chunk)} chars")

        return chunks

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.embedding_model.encode(text).tolist()
    
    def get(self, query: str, top_k: int = 3) -> Optional[List[Dict]]:
        """
        Retrieve relevant documents/chunks for the query
        Returns list of relevant documents if similarity exceeds threshold
        """
        debug_print(f"Searching RAG cache for query: '{query[:100]}...'")
        debug_print(f"Top-k: {top_k}, Threshold: {settings.rag_similarity_threshold}")

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
                for idx, result in enumerate(search_results):
                    metadata = result.payload.get("metadata", {})
                    doc_info = {
                        "content": result.payload.get("content"),
                        "metadata": metadata,
                        "score": result.score
                    }
                    documents.append(doc_info)

                    # Debug chunk information
                    if metadata.get("is_chunked"):
                        debug_print(f"  Result {idx+1}: chunk {metadata.get('chunk_index', 0)+1}/{metadata.get('total_chunks', 1)} "
                                  f"from doc {metadata.get('parent_doc_id', 'unknown')[:8]}, score={result.score:.4f}")
                    else:
                        debug_print(f"  Result {idx+1}: full document, score={result.score:.4f}")

                print(f"✓ Layer 2 (RAG Cache) HIT - Found {len(documents)} relevant documents")
                return documents

            print(f"✗ Layer 2 (RAG Cache) MISS")
            debug_print(f"No results above threshold {settings.rag_similarity_threshold}")
            return None
        except Exception as e:
            print(f"Error searching RAG cache: {e}")
            debug_print(f"Error details: {str(e)}")
            return None
    
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the RAG cache with optional chunking"""
        doc_id = str(uuid.uuid4())
        debug_print(f"Adding document with ID: {doc_id}")
        debug_print(f"Original content length: {len(content)} chars, {self._token_length(content)} tokens")

        # Split content into chunks
        chunks = self._chunk_text(content)

        try:
            points = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}" if len(chunks) > 1 else doc_id
                chunk_vector = self._embed_text(chunk)

                chunk_metadata = {
                    "parent_doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_chunked": len(chunks) > 1,
                    **(metadata or {})
                }

                point = PointStruct(
                    id=chunk_id,
                    vector=chunk_vector,
                    payload={
                        "content": chunk,
                        "metadata": chunk_metadata,
                        "timestamp": time.time()
                    }
                )
                points.append(point)
                debug_print(f"Created chunk {i+1}/{len(chunks)} with ID: {chunk_id}")

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            if len(chunks) > 1:
                print(f"✓ Added document to Layer 2 (RAG Cache) - ID: {doc_id} ({len(chunks)} chunks)")
            else:
                print(f"✓ Added document to Layer 2 (RAG Cache) - ID: {doc_id}")

            return doc_id
        except Exception as e:
            print(f"Error adding document to RAG cache: {e}")
            debug_print(f"Error details: {str(e)}")
            return None
    
    def add_documents_batch(self, documents: List[Dict]) -> List[str]:
        """
        Add multiple documents to RAG cache in batch with optional chunking
        documents: List of dicts with 'content' and optional 'metadata'
        """
        points = []
        doc_ids = []
        total_chunks = 0

        debug_print(f"Processing batch of {len(documents)} documents")

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content")
            metadata = doc.get("metadata", {})

            if not content:
                debug_print(f"Skipping document {doc_idx} - no content")
                continue

            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)

            # Split content into chunks
            chunks = self._chunk_text(content)
            total_chunks += len(chunks)

            debug_print(f"Document {doc_idx+1}/{len(documents)}: ID={doc_id}, chunks={len(chunks)}")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}" if len(chunks) > 1 else doc_id
                chunk_vector = self._embed_text(chunk)

                chunk_metadata = {
                    "parent_doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_chunked": len(chunks) > 1,
                    **metadata
                }

                points.append(PointStruct(
                    id=chunk_id,
                    vector=chunk_vector,
                    payload={
                        "content": chunk,
                        "metadata": chunk_metadata,
                        "timestamp": time.time()
                    }
                ))

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✓ Added {len(doc_ids)} documents ({total_chunks} chunks) to Layer 2 (RAG Cache)")
            debug_print(f"Successfully upserted {len(points)} points to Qdrant")
            return doc_ids
        except Exception as e:
            print(f"Error adding documents to RAG cache: {e}")
            debug_print(f"Error details: {str(e)}")
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

