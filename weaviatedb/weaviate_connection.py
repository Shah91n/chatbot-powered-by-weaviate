"""
Weaviate Connection Manager - Singleton Pattern
===============================================

This module provides a singleton Weaviate client manager for consistent connection
management across the entire application.

Usage:
    from weaviatedb.weaviate_connection import get_weaviate_client
    
    # For async operations (search, query)
    async def some_async_function():
        manager = get_weaviate_client()
        async with await manager.get_async_client() as client:
            result = await client.collections.get("my_collection").query.fetch_objects()
    
    # For sync operations (batch inserts)
    def some_sync_function():
        manager = get_weaviate_client()
        with manager.get_sync_client() as client:
            client.collections.get("my_collection").data.insert(...)
"""
import os
import logging
import contextlib
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class WeaviateConnectionManager(metaclass=Singleton):

    def __init__(self):
        load_dotenv()
        self._cluster_url = os.getenv("CLUSTER_URL") or None
        self._api_key = os.getenv("API_KEY") or None
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._headers = {}
        if self._openai_api_key:
            self._headers["X-OpenAI-Api-Key"] = self._openai_api_key
        self._weaviate_timeout = Timeout(init=30, query=120, insert=240)

    def _cloud_auth(self):
        return Auth.api_key(self._api_key) if self._api_key else None

    def get_async_client(self):
        """Return an async context manager for a Weaviate async client."""
        if not self._cluster_url:
            raise RuntimeError("CLUSTER_URL must be set for Weaviate Cloud.")
        return weaviate.use_async_with_weaviate_cloud(
            cluster_url=self._cluster_url,
            auth_credentials=self._cloud_auth(),
            headers=self._headers or None,
            additional_config=AdditionalConfig(timeout=self._weaviate_timeout),
        )

    @contextlib.contextmanager
    def get_sync_client(self):
        """Context manager that yields a synchronous Weaviate client."""
        if not self._cluster_url:
            raise RuntimeError("CLUSTER_URL must be set for Weaviate Cloud usage.")
        client = None
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._cluster_url,
                auth_credentials=self._cloud_auth(),
                headers=self._headers or None,
                additional_config=AdditionalConfig(timeout=self._weaviate_timeout),
            )
            yield client
        finally:
            try:
                if getattr(client, "close", None):
                    client.close()
            except Exception:
                logger.debug("Error closing sync client", exc_info=True)


def get_weaviate_client() -> WeaviateConnectionManager:
    """Return the singleton WeaviateConnectionManager instance."""
    return WeaviateConnectionManager()
