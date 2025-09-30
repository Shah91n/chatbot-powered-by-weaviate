import logging
from weaviate import WeaviateClient

from weaviate.classes.config import Configure, Property, DataType, Tokenization

logger = logging.getLogger(__name__)

# Schema configuration
SCHEMA_PROPERTIES = [
    Property(
        name="content",
        data_type=DataType.TEXT,
        description="The main text content of the document chunk",
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
        vectorize_property_name=False
    ),
    Property(
        name="fileName",
        data_type=DataType.TEXT,
        description="Original filename of the document",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=True,
        skip_vectorization=True
    ),
    Property(
        name="sourceFolder",
        data_type=DataType.TEXT,
        description="Source folder category",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=True,
        skip_vectorization=True
    ),
    Property(
        name="chunkIndex",
        data_type=DataType.INT,
        description="Index of this chunk within the document"
    ),
]

def create_schema(client: WeaviateClient, collection_name: str = "documents") -> bool:
    """
    Create or verify the schema for the documents collection using synchronous client.
    
    Args:
        client: Connected Weaviate sync client
        collection_name: Name of the collection to create/verify
        
    Returns:
        bool: True if schema exists or was created successfully
    """
    try:
        print("Creating schema...")
        
        # Check if collection already exists
        collection = client.collections.get(collection_name)
        if collection.exists():
            print(f"Collection '{collection_name}' already exists")
            return True
        
        # Create the collection with vectorizer configuration
        client.collections.create(
            name=collection_name,
            description="Document chunks for semantic search and retrieval",
            properties=SCHEMA_PROPERTIES,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small",
                vectorize_collection_name=False
            ),
            generative_config=Configure.Generative.openai()
        )
        
        print(f"✅ Collection '{collection_name}' created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        print(f"❌ Schema creation error: {e}")
        raise


def delete_schema(client: WeaviateClient, collection_name: str = "documents") -> bool:
    """
    Delete the specified collection/schema using synchronous client.
    
    Args:
        client: Connected Weaviate sync client
        collection_name: Name of the collection to delete
        
    Returns:
        bool: True if deletion was successful
    """
    try:
        print("Deleting schema...")
        collection = client.collections.get(collection_name)
        
        if not collection.exists():
            print(f"Collection '{collection_name}' does not exist")
            return True
            
        # Delete the collection
        client.collections.delete(collection_name)
        print(f"✅ Collection '{collection_name}' deleted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Schema deletion failed: {e}")
        print(f"❌ Schema deletion error: {e}")
        return False
