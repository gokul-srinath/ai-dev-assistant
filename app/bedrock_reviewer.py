import boto3
import os

from app.embedder import embed_text
# from app.qdrant_store import search_chunks
from app.qdrant_store import search_prd

import asyncio

from app.retriever import hybrid_search

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


async def review_patch(filename: str, patch: str, pr_number: int = None) -> str:
    # query_embedding = await embed_text(patch)
    pr_collection = f"code_chunks_pr_{pr_number}"

    main_chunks = await hybrid_search(patch, top_k=3, collection_name="code_chunks")
    pr_chunks = await hybrid_search(patch, top_k=3, collection_name=pr_collection)

    all_chunks = main_chunks + pr_chunks

    context = "\n\n".join(
        f"--- {c['filename']} (lines {c['start_line']}-{c['end_line']}) ---\n{c['content']}"
        for c in all_chunks
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


async def review_pr(files, pr_number):
    tasks = [
        review_patch(f["filename"], f["patch"], pr_number)
        for f in files if f.get("patch")
    ]
    results = await asyncio.gather(*tasks)
    return [
        {"filename": f["filename"], "comment": comment}
        for f, comment in zip(files, results)
        if f.get("patch")
    ]


async def validate_prd(files_content: dict, reviews: list[dict]) -> str:
    # use the review text as query to find relevant PRD sections
    query = "\n".join(r["comment"] for r in reviews)
    query_embedding = await embed_text(query)
    relevant_prd_chunks = await search_prd(query_embedding, top_k=5)

    prd_context = "\n\n".join(c["content"] for c in relevant_prd_chunks)

    combined_code = "\n\n".join(
        f"### {filename}\n```\n{content}\n```"
        for filename, content in files_content.items()
    )

    system_prompt = """You are a technical product manager validating a pull request against PRD requirements.
Rules:
- Only the actual file contents are source of truth
- Comments do not count as implementation
- If a requirement is partially met mark it NOT MET
- Met or Not Met only, no partial credit
- Do not reference what was previously there, only what exists now"""

    user_message = (
        f"## Relevant PRD Sections\n{prd_context}\n\n"
        f"## Actual PR File Contents\n{combined_code}"
    )

    response = await asyncio.to_thread(
        bedrock.converse,
        modelId=BEDROCK_MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"temperature": 0.1},
    )

    return response["output"]["message"]["content"][0]["text"]