"""
RAG Pipeline Doctor — FastAPI Server
Exposes OpenEnv-compliant HTTP endpoints: /reset  /step  /state  /health
"""

import sys
import os

# Ensure parent dir is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import RAGAction, RAGStepResult, RAGState
from server.rag_environment import RAGPipelineDoctorEnvironment


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline Doctor",
    description="OpenEnv RL environment: diagnose and repair broken RAG pipelines.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (for HF Spaces single-session use)
_env = RAGPipelineDoctorEnvironment()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None   # Pin a specific task (for eval); None = random


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "rag_pipeline_doctor", "version": "1.0.0"}


@app.post("/reset", response_model=RAGStepResult)
def reset(req: ResetRequest = ResetRequest()):
    result = _env.reset(task_id=req.task_id)
    return result


@app.post("/step", response_model=RAGStepResult)
def step(action: RAGAction):
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=RAGState)
def state():
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Web UI — interactive inspection at /
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "RAG Pipeline Doctor",
        "description": "OpenEnv RL environment for diagnosing and fixing broken RAG pipelines.",
        "endpoints": {
            "POST /reset": "Start new episode (optional: body {'task_id': 'chunking_error_001'})",
            "POST /step":  "Submit action {'diagnosis': '...', 'config_patch': {...}, 'confidence': 0.9}",
            "GET  /state": "Get current episode state",
            "GET  /health": "Health check",
            "GET  /docs":  "Swagger UI",
        },
        "tasks": [
            {"id": "chunking_error_001",         "difficulty": "easy"},
            {"id": "embedding_mismatch_001",     "difficulty": "medium"},
            {"id": "hallucination_retrieval_001","difficulty": "hard"},
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
