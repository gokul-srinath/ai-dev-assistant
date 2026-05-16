from qdrant_client import AsyncQdrantClient
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

# def get_collection_name(pr_number: int = None) -> str:
#     if pr_number:
#         return f"code_chunks_pr_{pr_number}"
#     return COLLECTION_NAME

async def delete_collection(collection_name: str):
    await client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted")

async def delete_chunks_by_filename(filename: str, collection_name: str = COLLECTION_NAME):
    await client.delete(
        collection_name=collection_name,
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

    
async def init_collection(collection_name: str = "code_chunks"):
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if collection_name not in names:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created")


async def store_chunks(chunks: list[dict], embeddings: list[list[float]], collection_name: str = COLLECTION_NAME):
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
        collection_name=collection_name,
        points=points
    )
    print(f"Stored {len(points)} chunks in Qdrant")


async def search_chunks(query_embedding: list[float], top_k: int = 5, collection_name: str = COLLECTION_NAME) -> list[dict]:
    results = await client.query_points(
        collection_name=collection_name,
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