from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["content"]) for c in chunks]
    print(pairs)
    scores = model.predict(pairs)
    print(scores)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]