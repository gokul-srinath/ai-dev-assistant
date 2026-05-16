from openai import AsyncOpenAI
import os

from app.embedder import embed_text
from app.qdrant_store import search_chunks

client = AsyncOpenAI(
    base_url=os.getenv("LM_STUDIO_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY")
)

LLM_MODEL = os.getenv("LLM_MODEL", "gemma-3-4b-it")

SYSTEM_PROMPT = """You are an expert code reviewer. You will be given:
1. A code diff (patch) from a pull request.
2. Relevant existing code from the codebase for context.

Provide a concise, actionable review covering:
- Bugs or logic errors
- Security concerns
- Performance issues
- Readability and maintainability improvements
- Naming or style inconsistencies with the existing codebase

Be specific — reference line numbers and variable names. If the change looks good, say so briefly. Do not repeat the code back."""


async def review_patch(filename: str, patch: str, pr_number: int = None) -> str:
    query_embedding = await embed_text(patch)
    pr_collection = f"code_chunks_pr_{pr_number}"


    # search main index for existing codebase context
    main_chunks = await search_chunks(query_embedding, top_k=3, collection_name="code_chunks")
    # search PR index for changed file context
    pr_chunks = await search_chunks(query_embedding, top_k=3, collection_name=pr_collection)

    all_chunks = main_chunks + pr_chunks

    context = "\n\n".join(
        f"--- {c['filename']} (lines {c['start_line']}-{c['end_line']}) ---\n{c['content']}"
        for c in all_chunks
    )

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## File: {filename}\n\n"
                    f"### Diff\n```\n{patch}\n```\n\n"
                    f"### Relevant codebase context\n{context}"
                ),
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


async def review_pr(files: list[dict], pr_number: int) -> list[dict]:
    reviews = []
    for f in files:
        if not f.get("patch"):
            continue
        comment = await review_patch(f["filename"], f["patch"], pr_number)
        reviews.append({"filename": f["filename"], "comment": comment})
    return reviews
