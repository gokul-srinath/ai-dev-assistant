import boto3
import json
import os

from app.embedder import embed_text
from app.qdrant_store import search_chunks
import asyncio

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

BEDROCK_MODEL = os.getenv("BEDROCK_LLM_MODEL", "apac.amazon.nova-micro-v1:0")

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


async def review_patch(filename: str, patch: str) -> str:
    query_embedding = await embed_text(patch)
    relevant_chunks = await search_chunks(query_embedding, top_k=5)

    context = "\n\n".join(
        f"--- {c['filename']} (lines {c['start_line']}-{c['end_line']}) ---\n{c['content']}"
        for c in relevant_chunks
    )

    user_message = (
        f"## File: {filename}\n\n"
        f"### Diff\n```\n{patch}\n```\n\n"
        f"### Relevant codebase context\n{context}"
    )

    response = await asyncio.to_thread(bedrock.converse,
        modelId=BEDROCK_MODEL,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {"role": "user", "content": [{"text": user_message}]},
        ],
        inferenceConfig={"temperature": 0.3},
    )

    return response["output"]["message"]["content"][0]["text"]


async def review_pr(files: list[dict]) -> list[dict]:
    reviews = []
    for f in files:
        if not f.get("patch"):
            continue
        comment = await review_patch(f["filename"], f["patch"])
        reviews.append({"filename": f["filename"], "comment": comment})
    return reviews
