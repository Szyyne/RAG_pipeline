"""
RAG Pipeline Doctor — Typed Models
Action / Observation / State for the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class RAGAction(BaseModel):
    """
    The agent submits a fix as a JSON patch to the pipeline config,
    along with a diagnosis string explaining the root cause.
    """
    diagnosis: str                          # Agent's explanation of what's wrong
    config_patch: Dict[str, Any]            # Key-value fixes to apply to pipeline config
    confidence: float = 1.0                 # 0.0–1.0 self-reported confidence


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class RetrievalSample(BaseModel):
    query: str
    retrieved_chunks: List[str]
    expected_answer: str
    actual_answer: str                      # Answer produced by broken pipeline
    retrieval_score: float                  # 0.0–1.0 NDCG-style score


class RAGObservation(BaseModel):
    """Everything the agent sees each step."""
    task_id: str                            # e.g. "chunking_error_001"
    difficulty: str                         # easy / medium / hard
    pipeline_config: Dict[str, Any]         # Current (possibly broken) config
    failure_symptoms: List[str]             # Observable symptoms (e.g. "answers too short")
    retrieval_samples: List[RetrievalSample]  # 3 sample query→retrieve→answer triplets
    step_number: int
    max_steps: int
    last_action_feedback: Optional[str] = None   # Feedback on previous fix attempt
    solved: bool = False


# ---------------------------------------------------------------------------
# State  (episode metadata)
# ---------------------------------------------------------------------------

class RAGState(BaseModel):
    episode_id: str
    task_id: str
    difficulty: str
    total_steps: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    diagnosis_score: float = 0.0
    fix_score: float = 0.0
    faithfulness_score: float = 0.0


# ---------------------------------------------------------------------------
# StepResult (returned by step() / reset())
# ---------------------------------------------------------------------------

class RAGStepResult(BaseModel):
    observation: RAGObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}
