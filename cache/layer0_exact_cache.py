import hashlib
import json
import redis
from typing import Optional
from config import settings, debug_print


class ExactCache:
    """
    Layer 0: Exact Cache using Redis
    Provides fast exact match lookups for queries
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True
        )
    
    def _generate_key(self, query: str) -> str:
        """Generate a unique key for the query using hash"""
        return f"exact_cache:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get(self, query: str) -> Optional[str]:
        """Retrieve cached response for exact query match"""
        key = self._generate_key(query)
        debug_print(f"Layer 0: Looking up key '{key}'")
        cached_value = self.redis_client.get(key)

        if cached_value:
            print(f"✓ Layer 0 (Exact Cache) HIT for query: {query[:50]}...")
            debug_print(f"Layer 0: Retrieved {len(cached_value)} chars from cache")
            return cached_value

        print(f"✗ Layer 0 (Exact Cache) MISS")
        debug_print(f"Layer 0: No cached value found for key '{key}'")
        return None
    
    def set(self, query: str, response: str, ttl: Optional[int] = None) -> None:
        """Store response in exact cache"""
        key = self._generate_key(query)
        ttl = ttl or settings.cache_ttl
        debug_print(f"Layer 0: Storing {len(response)} chars with TTL={ttl}s at key '{key}'")
        self.redis_client.setex(key, ttl, response)
        print(f"✓ Stored in Layer 0 (Exact Cache)")
    
    def delete(self, query: str) -> None:
        """Delete cached response"""
        key = self._generate_key(query)
        self.redis_client.delete(key)
    
    def clear_all(self) -> None:
        """Clear all exact cache entries"""
        count = 0
        for key in self.redis_client.scan_iter("exact_cache:*"):
            self.redis_client.delete(key)
            count += 1
        debug_print(f"Layer 0: Deleted {count} cache entries")
        print("✓ Cleared all Layer 0 (Exact Cache) entries")
    
    def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            print(f"Redis health check failed: {e}")
            return False

