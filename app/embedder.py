from openai import AsyncOpenAI
import os


client = AsyncOpenAI(
    base_url=os.getenv("LM_STUDIO_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")

async def embed_text(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding