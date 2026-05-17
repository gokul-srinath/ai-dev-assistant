from app.bm25_store import bm25_index
from app.reranker import rerank
from app.qdrant_store import search_chunks
from app.embedder import embed_text

async def hybrid_search(query: str, top_k: int = 5, collection_name: str = "code_chunks") -> list[dict]:
    # vector search
    query_embedding = await embed_text(query)
    vector_results = await search_chunks(query_embedding, top_k=top_k, collection_name=collection_name)

    # BM25 search
    bm25_results = bm25_index.search(query, top_k=top_k)

    # combine — deduplicate by filename + start_line
    seen = set()
    combined = []
    for chunk in vector_results + bm25_results:
        key = (chunk["filename"], chunk["start_line"])
        if key not in seen:
            seen.add(key)
            combined.append(chunk)

    # rerank combined results
    reranked = rerank(query, combined, top_k=top_k)
    return reranked