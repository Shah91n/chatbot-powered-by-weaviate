from typing import List, Callable
import weaviate
from weaviate.util import generate_uuid5

def ingest_chunks_to_weaviate(client: weaviate.WeaviateClient, chunks: List[dict], collection_name: str,
                              update_progress_callback: Callable[[float], None] = None,
                              log_callback: Callable[[str], None] = None):
    """Batch ingest chunks into Weaviate. Returns dict with counts and failed objects."""

    collection = client.collections.use(collection_name)
    successful = 0
    failed = 0

    try:
        if log_callback:
            log_callback(f"Starting batch ingest: {len(chunks)} items")

        with collection.batch.fixed_size() as batch:
            for i, chunk in enumerate(chunks):
                properties = {
                    "content": chunk["content"],
                    "fileName": chunk["fileName"],
                    "sourceFolder": chunk.get("sourceFolder", "uploads"),
                    "chunkIndex": chunk.get("chunkIndex", i),
                }

                # deterministic uuid
                uuid = generate_uuid5(f"{properties['fileName']}_{properties['sourceFolder']}_{properties['chunkIndex']}")

                batch.add_object(properties=properties, uuid=uuid)
                if log_callback:
                    log_callback(f"Queued object {i+1}/{len(chunks)} (file={properties['fileName']}, chunk={properties['chunkIndex']})")
                successful += 1

                if update_progress_callback:
                    update_progress_callback((i + 1) / len(chunks))

        # After closing batch, check for failures
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            failed = len(failed_objects)
            return {"successful": successful - failed, "failed": failed, "failed_objects": failed_objects}

        return {"successful": successful, "failed": 0, "failed_objects": []}

    except Exception as e:
        # keep minimal but useful error info
        return {"successful": successful, "failed": len(chunks) - successful, "error": str(e)}
