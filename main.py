from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import asyncio
import os
import base64
from datetime import datetime
from github import Github, GithubException
from huggingface_hub import HfApi, HfFolder
from openai import OpenAI
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="AI Project Generator")

# Configuration
VALID_SECRET = os.getenv("VALID_SECRET", "default-secret")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AIPIPE_TOKEN")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
JULEP_API_KEY = os.getenv("JULEP_API_KEY")
JULEP_BASE_URL = os.getenv("JULEP_BASE_URL", "https://api.julep.ai/v1")

# Initialize clients
gh = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Pydantic models
class Attachment(BaseModel):
    name: str
    url: str
    content: Optional[str] = None

class Round1Request(BaseModel):
    secret: str
    email: str
    task: str
    brief: str
    nonce: str
    round: int = 1
    evaluation_url: str
    attachments: List[Attachment] = []

class Round2Request(BaseModel):
    secret: str
    email: str
    task: str
    brief: str
    nonce: str
    round: int = 2
    evaluation_url: str

class CodeGenerationRequest(BaseModel):
    brief: str
    attachments: Optional[List[Dict[str, str]]] = None
    current_code: Optional[str] = None
    round: int = 1

# Helper functions
async def verify_secret(secret: str) -> bool:
    """Verify request secret"""
    return secret == VALID_SECRET

async def fetch_attachment_data(url: str) -> str:
    """Fetch attachment content"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            if url.startswith("data:"):
                # Handle data URLs
                return url.split(",")[1]
            return base64.b64encode(response.content).decode()
    except Exception as e:
        logger.error(f"Error fetching attachment {url}: {e}")
        return ""

async def generate_app_code_with_llm(brief: str, attachments: Optional[List] = None, current_code: Optional[str] = None, round: int = 1) -> str:
    """Generate app code using LLM"""
    
    attachment_context = ""
    if attachments:
        attachment_context = "Attachments available:\n" + "\n".join([f"- {a.get('name', 'unknown')}" for a in attachments])
    
    modify_instruction = ""
    if round > 1 and current_code:
        modify_instruction = f"""
Current code to modify/improve:
```html
{current_code[:2000]}...
```

Maintain compatibility while improving."""

    prompt = f"""You are an expert web developer. Generate a complete, production-ready single-page HTML/CSS/JS application.

TASK BRIEF:
{brief}

{attachment_context}

{modify_instruction}

REQUIREMENTS:
1. Return ONLY valid HTML (single file with inline CSS/JS)
2. Must be runnable immediately when opened
3. Use fetch() for data loading from attachments
4. Include comprehensive error handling
5. Make it responsive and accessible
6. Include Bootstrap 5 or Tailwind if mentioned
7. All business logic must work client-side
8. If round {round} > 1, improve and extend features

CRITICAL: Return ONLY the HTML code with no explanations or markdown."""

    try:
        # Use OpenAI GPT-4o-mini via AIPipe
        message = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        # Fallback minimal app
        return f"""<!DOCTYPE html>
<html>
<head><title>{brief[:30]}</title></head>
<body><h1>App Generated</h1><p>{brief}</p></body>
</html>"""

async def create_github_repo(email: str, task: str, app_code: str, attachments: Optional[List] = None) -> Dict[str, str]:
    """Create GitHub repo and push code"""
    if not gh:
        raise HTTPException(status_code=500, detail="GitHub not configured")
    
    try:
        user = gh.get_user()
        repo_name = task.lower().replace("_", "-").replace(" ", "-")[:50]
        repo_name = f"{repo_name}-{datetime.now().strftime('%s')[-6:]}"
        
        # Create repo
        repo = user.create_repo(
            name=repo_name,
            description=f"Auto-generated app for: {task}",
            private=False,
            auto_init=True
        )
        logger.info(f"Created repo: {repo.html_url}")
        
        # Push index.html
        repo.create_file_contents(
            path="index.html",
            message="Add generated app code",
            content=app_code
        )
        
        # Push LICENSE
        license_content = f"""MIT License

Copyright (c) {datetime.now().year} Generated App

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software."""
        
        repo.create_file_contents(
            path="LICENSE",
            message="Add MIT License",
            content=license_content
        )
        
        # Create README
        readme_content = f"""# {task}

## Summary
Auto-generated single-page application created via AI Project Generator.

## Setup
Clone and open `index.html` in your browser.

## Usage
Visit the live deployment at: https://{user.login}.github.io/{repo_name}/

## About
This app was auto-generated using AI based on the following brief:
> {brief[:200]}

## Features
- Responsive design
- Client-side processing
- Error handling
- Accessible markup

## License
MIT License - see LICENSE file for details."""
        
        repo.create_file_contents(
            path="README.md",
            message="Add README",
            content=readme_content
        )
        
        # Update homepage for GitHub Pages
        repo.edit(homepage=f"https://{user.login}.github.io/{repo_name}/")
        
        # Get commit SHA
        commits = repo.get_commits()
        commit_sha = commits[0].sha
        
        return {
            "repo_url": repo.html_url,
            "commit_sha": commit_sha,
            "pages_url": f"https://{user.login}.github.io/{repo_name}/",
            "repo_name": repo_name
        }
    except GithubException as e:
        logger.error(f"GitHub error: {e}")
        raise HTTPException(status_code=500, detail=f"GitHub error: {str(e)}")

async def update_github_repo(email: str, task: str, brief: str) -> Dict[str, str]:
    """Update existing GitHub repo for round 2"""
    if not gh:
        raise HTTPException(status_code=500, detail="GitHub not configured")
    
    try:
        user = gh.get_user()
        repos = user.get_repos()
        
        # Find matching repo
        target_repo = None
        task_clean = task.lower().replace("_", "-").replace(" ", "-")
        for repo in repos:
            if task_clean in repo.name.lower():
                target_repo = repo
                break
        
        if not target_repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Fetch current code
        current_file = target_repo.get_contents("index.html")
        current_code = current_file.decoded_content.decode()
        
        # Generate updated code
        updated_code = await generate_app_code_with_llm(
            brief=f"{brief}\n\nImproving existing app.",
            current_code=current_code,
            round=2
        )
        
        # Update files
        target_repo.update_file_contents(
            path="index.html",
            message=f"Round 2 Update: {brief[:50]}",
            content=updated_code,
            sha=current_file.sha
        )
        
        # Update README
        readme_file = target_repo.get_contents("README.md")
        updated_readme = readme_file.decoded_content.decode() + f"\n\n### Round 2 Update\n{brief}"
        target_repo.update_file_contents(
            path="README.md",
            message=f"Update for round 2",
            content=updated_readme,
            sha=readme_file.sha
        )
        
        # Get new commit SHA
        commits = target_repo.get_commits()
        commit_sha = commits[0].sha
        
        return {
            "repo_url": target_repo.html_url,
            "commit_sha": commit_sha,
            "pages_url": f"https://{user.login}.github.io/{target_repo.name}/",
            "repo_name": target_repo.name
        }
    except Exception as e:
        logger.error(f"Update repo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def push_results_with_retry(evaluation_url: str, payload: Dict[str, Any], max_retries: int = 8) -> bool:
    """Push results to evaluation URL with exponential backoff"""
    delay = 1
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    evaluation_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    logger.info(f"Successfully posted to {evaluation_url}")
                    return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
    
    logger.error(f"Failed to post after {max_retries} retries")
    return False

# Background task
async def process_round1(req: Round1Request):
    """Process Round 1 request"""
    try:
        logger.info(f"Processing Round 1 for task: {req.task}")
        
        # Fetch attachments if provided
        processed_attachments = []
        if req.attachments:
            for att in req.attachments:
                content = await fetch_attachment_data(att.url)
                processed_attachments.append({"name": att.name, "content": content})
        
        # Generate app code
        app_code = await generate_app_code_with_llm(
            brief=req.brief,
            attachments=processed_attachments,
            round=1
        )
        
        # Create GitHub repo
        repo_info = await create_github_repo(req.email, req.task, app_code, processed_attachments)
        
        # Prepare evaluation payload
        payload = {
            "email": req.email,
            "task": req.task,
            "round": req.round,
            "nonce": req.nonce,
            "repo_url": repo_info["repo_url"],
            "commit_sha": repo_info["commit_sha"],
            "pages_url": repo_info["pages_url"]
        }
        
        # Push results with retry
        await push_results_with_retry(req.evaluation_url, payload)
        logger.info(f"Round 1 completed for {req.task}")
    except Exception as e:
        logger.error(f"Round 1 processing error: {e}")

async def process_round2(req: Round2Request):
    """Process Round 2 request"""
    try:
        logger.info(f"Processing Round 2 for task: {req.task}")
        
        # Update GitHub repo
        repo_info = await update_github_repo(req.email, req.task, req.brief)
        
        # Prepare evaluation payload
        payload = {
            "email": req.email,
            "task": req.task,
            "round": req.round,
            "nonce": req.nonce,
            "repo_url": repo_info["repo_url"],
            "commit_sha": repo_info["commit_sha"],
            "pages_url": repo_info["pages_url"]
        }
        
        # Push results with retry
        await push_results_with_retry(req.evaluation_url, payload)
        logger.info(f"Round 2 completed for {req.task}")
    except Exception as e:
        logger.error(f"Round 2 processing error: {e}")

# Routes
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "github": "✓" if gh else "✗",
        "hugging_face": "✓" if hf_api else "✗",
        "llm": "✓" if OPENAI_API_KEY else "✗"
    }

@app.post("/api-endpoint")
async def round1_handler(req: Round1Request, background_tasks: BackgroundTasks):
    """Round 1: Generate and deploy app"""
    try:
        # Verify secret
        if not await verify_secret(req.secret):
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        # Immediate response
        background_tasks.add_task(process_round1, req)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "Processing",
                "message": "Request received and processing",
                "task": req.task,
                "round": 1
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Round 1 handler error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api-endpoint/round2")
async def round2_handler(req: Round2Request, background_tasks: BackgroundTasks):
    """Round 2: Update and enhance app"""
    try:
        # Verify secret
        if not await verify_secret(req.secret):
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        # Immediate response
        background_tasks.add_task(process_round2, req)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "Processing",
                "message": "Round 2 request received and processing",
                "task": req.task,
                "round": 2
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Round 2 handler error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-code")
async def generate_code(req: CodeGenerationRequest):
    """Direct code generation endpoint"""
    try:
        code = await generate_app_code_with_llm(
            brief=req.brief,
            attachments=req.attachments,
            current_code=req.current_code,
            round=req.round
        )
        return JSONResponse(
            status_code=200,
            content={"code": code, "status": "success"}
        )
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI Project Generator",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "round1": "POST /api-endpoint",
            "round2": "POST /api-endpoint/round2",
            "generate": "POST /generate-code"
        }
    }

# Vercel serverless entry
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)