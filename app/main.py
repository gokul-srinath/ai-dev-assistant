from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from app.embedder import embed_text
from app.github import get_changed_files, get_file_content, post_pr_comment
from app.parser import extract_chunks
from app.qdrant_store import delete_chunks_by_filename, store_chunks, init_collection
from app.bedrock_reviewer import review_pr as bedrock_reviewer_pr
import hmac, hashlib, os



WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_collection()
    yield


app = FastAPI(lifespan=lifespan)

def verify_signature(payload: bytes, signature: str) -> bool:
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")

    if not verify_signature(payload, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()
    action = data.get("action")

    if action in ["opened", "synchronize"]:
        pr_number = data["pull_request"]["number"]
        repo_name = data["repository"]["full_name"]

        print(f"PR #{pr_number} opened in {repo_name}")

        files = await get_changed_files(repo_name, pr_number)
        all_chunks = []
        for f in files:
            content = await get_file_content(f["raw_url"])
            chunks = extract_chunks(f["filename"], content)
            all_chunks.extend(chunks)
            # print(f"--- {f['filename']} -> {len(chunks)} chunks ---")

        for f in files:
            await delete_chunks_by_filename(f["filename"])
            print(f"Deleted existing chunks for {f['filename']}")

        embeddings = []
        for chunk in all_chunks:
            embedding = await embed_text(chunk["content"])
            embeddings.append(embedding)
        
        await store_chunks(all_chunks, embeddings)

        reviews = await bedrock_reviewer_pr(files)
        
        body = "\n\n---\n\n".join(
            f"### `{r['filename']}`\n{r['comment']}"
            for r in reviews
        )
        await post_pr_comment(repo_name, pr_number, body)
        print(f"Comment posted to PR #{pr_number}")

    return {"status": "ok"}