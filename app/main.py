from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from app.embedder import embed_text
from app.github import get_all_repo_files, get_changed_files, get_file_content, post_pr_comment
from app.parser import extract_chunks
from app.qdrant_store import delete_chunks_by_filename, delete_collection, store_chunks, init_collection
from app.bedrock_reviewer import review_pr as bedrock_reviewer_pr
import hmac, hashlib, os



WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

async def index_full_repo(repo_name: str, branch: str = "master"):
    files = await get_all_repo_files(repo_name, branch)
    print(f"Indexing {len(files)} files from {repo_name}")

    for f in files:
        try:
            content = await get_file_content(f["raw_url"])
            chunks = extract_chunks(f["filename"], content)
            if not chunks:
                continue
            await delete_chunks_by_filename(f["filename"])
            embeddings = [await embed_text(c["content"]) for c in chunks]
            await store_chunks(chunks, embeddings)
            print(f"Indexed {f['filename']} → {len(chunks)} chunks")
        except Exception as e:
            print(f"Skipped {f['filename']}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_collection()
    repo_name = os.getenv("REPO_NAME")
    branch = os.getenv("REPO_BRANCH","master")
    if(repo_name):
        await index_full_repo(repo_name, branch)
    else:
        print("No repo name provided")
    yield


app = FastAPI(lifespan=lifespan)

def verify_signature(payload: bytes, signature: str) -> bool:
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/index")
async def index_repo(repo_name: str, branch: str = "master"):
    await index_full_repo(repo_name, branch)
    return {"status": "indexed"}


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")

    if not verify_signature(payload, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()
    action = data.get("action")
    main_collection = "code_chunks"

    if action in ["opened", "synchronize"]:
        pr_number = data["pull_request"]["number"]
        repo_name = data["repository"]["full_name"]
        
        print(f"PR #{pr_number} {action.capitalize()} in {repo_name}")

        pr_collection = f"code_chunks_pr_{pr_number}"
        await init_collection(pr_collection)


        files = await get_changed_files(repo_name, pr_number)
        all_chunks = []
        for f in files:
            content = await get_file_content(f["raw_url"])
            chunks = extract_chunks(f["filename"], content)
            all_chunks.extend(chunks)
            # print(f"--- {f['filename']} -> {len(chunks)} chunks ---")

        embeddings = []
        for chunk in all_chunks:
            embedding = await embed_text(chunk["content"])
            embeddings.append(embedding)
        
        await store_chunks(all_chunks, embeddings, collection_name=pr_collection)
        try:
            reviews = await bedrock_reviewer_pr(files, pr_number)
        except Exception as e:
            print(f"Error reviewing PR #{pr_number}: {e}")
            reviews = []
        if(reviews):
            try:
                body = "\n\n---\n\n".join(
                    f"### `{r['filename']}`\n{r['comment']}"
                    for r in reviews
                )
                await post_pr_comment(repo_name, pr_number, f"## AI Code Review\n\n{body}")
                print(f"AI Code Review posted to PR #{pr_number}")
            except Exception as e:
                print(f"Error posting AI Code Review: {e}")

    elif action == "closed":    
        if(not data["pull_request"]["merged"]):
            return {"status": "PR not merged, skipping index update"}

        pr_number = data["pull_request"]["number"]
        repo_name = data["repository"]["full_name"]
        pr_collection = f"code_chunks_pr_{pr_number}"

        files = await get_changed_files(repo_name, pr_number)
        for f in files:
            content = await get_file_content(f["raw_url"])
            chunks = extract_chunks(f["filename"], content)
            await delete_chunks_by_filename(f["filename"], collection_name=main_collection)
            embeddings = []
            for chunk in chunks:
                embedding = await embed_text(chunk["content"])
                embeddings.append(embedding)
            await store_chunks(chunks, embeddings, collection_name=main_collection)
            print(f"Main index updated with merged PR #{pr_number}")
        #always delete the PR collection    
        await delete_collection(pr_collection)
    return {"status": "ok"}