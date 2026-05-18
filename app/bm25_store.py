from rank_bm25 import BM25Okapi
import re

class BM25Index:
    def __init__(self):
        self.chunks = []
        self.index = None

    def build(self, chunks: list[dict]):
        if not chunks:
            print("BM25: no chunks to index, skipping")
            return
        self.chunks = chunks
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.index:
            return []
        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self.chunks[i], "score": scores[i]}
            for i in top_indices
            if scores[i] > 0
        ]

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        # print(tokens)
        return tokens

bm25_index = BM25Index()