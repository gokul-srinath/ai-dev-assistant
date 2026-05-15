from fastapi import FastAPI, Request, HTTPException
from app.github import get_changed_files, get_file_content
import hmac, hashlib, os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

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
        for f in files:
            content = await get_file_content(f["raw_url"])
            print(f"--- {f['filename']} ---")
            print(content[:200])  # print first 200 chars

    return {"status": "ok"}