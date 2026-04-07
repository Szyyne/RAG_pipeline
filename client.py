"""
RAG Pipeline Doctor — Python Client
Mirrors the OpenEnv HTTPEnvClient interface.
Usage:
    with RAGPipelineDoctorEnv(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset()
        result = env.step(RAGAction(diagnosis="...", config_patch={...}))
        state  = env.state()
"""

import requests
from typing import Optional
from models import RAGAction, RAGStepResult, RAGState


class RAGPipelineDoctorEnv:
    """Synchronous HTTP client for the RAG Pipeline Doctor environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, task_id: Optional[str] = None) -> RAGStepResult:
        payload = {"task_id": task_id} if task_id else {}
        resp = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return RAGStepResult(**resp.json())

    def step(self, action: RAGAction) -> RAGStepResult:
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=30,
        )
        resp.raise_for_status()
        return RAGStepResult(**resp.json())

    def state(self) -> RAGState:
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return RAGState(**resp.json())

    def health(self) -> dict:
        resp = self._session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def sync(self):
        """Compatibility shim — returns self for use as context manager."""
        return self
