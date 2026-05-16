import httpx
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

async def get_changed_files(repo_name: str, pr_number: int):
    try:
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
        print(f"Changed files: {[x['filename'] for x in changed_files]}")
        return changed_files
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return []


async def get_file_content(raw_url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:

        response = await client.get(raw_url, headers=HEADERS)
        response.raise_for_status()
        return response.text

async def post_pr_comment(repo_name: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=HEADERS, json={"body": body})
        response.raise_for_status()
        print(f"Comment posted to PR #{pr_number}")