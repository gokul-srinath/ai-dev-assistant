import httpx
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

async def get_changed_files(repo_name: str, pr_number: int):
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}/files"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=HEADERS)
        response.raise_for_status()
        files = response.json()
    
    changed_files = []
    for f in files:
        if f["filename"].endswith((".py", ".js", ".ts", ".html", ".css", ".tsx", ".jsx")):
            changed_files.append({
                "filename": f["filename"],
                "patch": f.get("patch", ""),
                "raw_url": f["raw_url"]
            })
    
    return changed_files


async def get_file_content(raw_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(raw_url, headers=HEADERS)
        response.raise_for_status()
        return response.text