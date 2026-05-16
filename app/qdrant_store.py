from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)
import os
import uuid

client = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

COLLECTION_NAME = "code_chunks"
VECTOR_SIZE = 768


from qdrant_client.models import Filter, FieldCondition, MatchValue

async def delete_chunks_by_filename(filename: str):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename)
                )
            ]
        )
    )
    print(f"Deleted existing chunks for {filename}")

    
async def init_collection():
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]

    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")


async def store_chunks(chunks: list[dict], embeddings: list[list[float]]):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "filename": chunk["filename"],
                "type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["content"]
            }
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Stored {len(points)} chunks in Qdrant")


async def search_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
    query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": r.score,
            "filename": r.payload["filename"],
            "type": r.payload["type"],
            "start_line": r.payload["start_line"],
            "end_line": r.payload["end_line"],
            "content": r.payload["content"]
        }
        for r in results.points
    ]