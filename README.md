# 🩺 RAG Pipeline Doctor

> **OpenEnv Hackathon Submission** | Meta × PyTorch × Scaler  
> An RL environment where an AI agent diagnoses and repairs broken RAG pipelines.

---

## Overview

**RAG Pipeline Doctor** is a real-world OpenEnv environment built for the [OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/). 

The agent acts as a senior ML engineer parachuted into a broken production RAG system. It receives:
- A broken pipeline configuration
- Observable failure symptoms  
- Sample query → retrieval → answer triplets (with expected vs actual answers)

It must **diagnose the root cause** and **submit a config patch** to fix the pipeline — scored across diagnosis quality, fix correctness, and answer faithfulness.

### Why This Matters

RAG pipeline failures are one of the most common and costly issues in production LLM systems. Bugs are often subtle: a stale index silently returns outdated information, an embedding mismatch causes retrieval to completely miss, a chunk size that's too small truncates all context. This environment trains agents to catch and fix exactly these failure modes.

---

## Environment Description

### Action Space

The agent submits a structured action each step:

```json
{
  "diagnosis": "The chunk_size of 50 tokens is too small causing context truncation",
  "config_patch": {
    "chunk_size": 512,
    "chunk_overlap": 50
  },
  "confidence": 0.9
}
```

| Field | Type | Description |
|-------|------|-------------|
| `diagnosis` | `str` | Agent's natural language explanation of the root cause |
| `config_patch` | `dict` | Key-value fixes to apply to the broken pipeline config |
| `confidence` | `float` | Self-reported confidence score (0.0–1.0) |

### Observation Space

Each observation exposes:

```json
{
  "task_id": "chunking_error_001",
  "difficulty": "easy",
  "pipeline_config": { "chunk_size": 50, "chunk_overlap": 0, ... },
  "failure_symptoms": [
    "Answers are incomplete and cut off mid-sentence",
    "Retrieved chunks contain only 5-10 words each"
  ],
  "retrieval_samples": [
    {
      "query": "What are the key benefits of transformer architectures?",
      "retrieved_chunks": ["Transformers use self-att", "attention allows parallel", ...],
      "expected_answer": "Transformers use self-attention to process sequences in parallel...",
      "actual_answer": "Transformers use self-att... [truncated context]",
      "retrieval_score": 0.18
    }
  ],
  "step_number": 0,
  "max_steps": 5,
  "last_action_feedback": null,
  "solved": false
}
```

### State

```json
{
  "episode_id": "a3f8c1d2",
  "task_id": "embedding_mismatch_001",
  "difficulty": "medium",
  "total_steps": 2,
  "cumulative_reward": 0.43,
  "done": false,
  "diagnosis_score": 0.67,
  "fix_score": 0.50,
  "faithfulness_score": 0.48
}
```

---

## Tasks

### Task 1 — `chunking_error_001` | Easy

**Root cause:** `chunk_size` is set to 50 tokens (should be ~512). Every retrieved chunk is truncated to a few words, destroying all semantic content before it reaches the LLM.

**Observable signals:** Answers are cut off, retrieval scores are near 0.12–0.18, context window filled with fragments.

**What the agent must do:** Identify `chunk_size` as too small and patch it to a reasonable value (256–1024), optionally adding `chunk_overlap`.

**Grading:**
- Diagnosis score: % of key terms identified (chunk, size, truncation, overlap)
- Fix score: 1.0 if `256 ≤ chunk_size ≤ 1024`, partial otherwise
- Faithfulness: proxy of fix score (better config → better answers)

---

### Task 2 — `embedding_mismatch_001` | Medium

**Root cause:** The document index was built using `sentence-transformers/all-MiniLM-L6-v2` but queries are encoded using `BAAI/bge-large-en-v1.5`. The two models live in incompatible embedding spaces — cosine similarity between index and query vectors is essentially random.

**Observable signals:** Retrieved chunks are completely off-topic (e.g. asking about neural nets, getting chunks about the French Revolution). Similarity scores are 0.03–0.06. Increasing `top_k` does not help.

**What the agent must do:** Identify the embedding model mismatch and patch both `index_embedding_model` and `query_embedding_model` to the same model.

**Grading:**
- Fix score: 1.0 only if both models are patched to the same value

---

### Task 3 — `hallucination_retrieval_001` | Hard

**Root cause:** Four simultaneous bugs:
1. `similarity_metric: dot_product` (should be `cosine` for normalized BGE embeddings)
2. `index_version: v1.2` (stale — documents updated but index not rebuilt)
3. `reranker_enabled: false` (noisy top-k passes through unfiltered)
4. `context_window_strategy: first_k` (should be `most_relevant`)

**Why it's hard:** The failure is *silent*. Retrieval returns documents from the right domain, so the LLM produces confident-sounding answers that are factually wrong (e.g. citing outdated benchmark scores). Faithfulness score is a deceptive 0.45 — the pipeline appears to be working.

**What the agent must do:** Identify all four bugs. Partial credit awarded per fix.

**Grading:** Each of the 4 fixes is worth 0.25. Agent must diagnose multiple interacting issues.

---

## Reward Function

```
reward = 0.35 × diagnosis_score
       + 0.45 × fix_score
       + 0.20 × faithfulness_score
```

**Bonuses and penalties:**
- +0.15 bonus for solving the task on the **first attempt** (fix_score ≥ 0.9)
- −0.05 penalty for high-confidence wrong answers (confidence > 0.8 but fix_score < 0.3)

**Reward range:** 0.0 to 1.0 per step. Max 5 steps per episode.

The reward is decomposed rather than binary — the agent gets partial credit for a correct diagnosis even if the config patch is incomplete, encouraging it to at least reason correctly about failure modes.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Docker
- Hugging Face CLI (`pip install huggingface_hub`)

### Local Setup

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/rag-pipeline-doctor
cd rag-pipeline-doctor

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — smoke test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

### Run Tests

```bash
python tests/test_env.py
```

### Docker

```bash
docker build -t rag-pipeline-doctor .
docker run -p 7860:7860 rag-pipeline-doctor
```

---

## API Reference

### `POST /reset`

Start a new episode. Optionally pin a task for deterministic evaluation.

```bash
# Random task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

# Specific task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "embedding_mismatch_001"}'
```

### `POST /step`

Submit an action (diagnosis + config patch).

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "diagnosis": "chunk_size is 50 which truncates all context",
    "config_patch": {"chunk_size": 512, "chunk_overlap": 50},
    "confidence": 0.9
  }'
```

### `GET /state`

Get current episode metadata.

```bash
curl http://localhost:7860/state
```

### `GET /health`

```bash
curl http://localhost:7860/health
# {"status": "ok", "env": "rag_pipeline_doctor", "version": "1.0.0"}
```

---

## Running Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="https://YOUR_SPACE.hf.space"

python inference.py
```

### Mandatory Stdout Format

The script emits exactly three line types per episode — **deviation causes incorrect evaluation scoring**:

```
[START] task=chunking_error_001 env=rag_pipeline_doctor model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"patch":{"chunk_size":512}} reward=0.82 done=true error=null
[END] success=true steps=1 score=0.164 rewards=0.82
```

Rules:
- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after episode ends — always emitted, even on exception
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase: `true` or `false`
- `error` is the error string or `null`
- All fields on a single line, no newlines within a line

### Expected Baseline Scores (zero-shot Qwen2.5-72B)

| Task | Difficulty | Fix Score | Score |
|------|-----------|-----------|-------|
| chunking_error_001 | Easy | ~0.70 | ~0.164 |
| embedding_mismatch_001 | Medium | ~0.50 | ~0.130 |
| hallucination_retrieval_001 | Hard | ~0.25 | ~0.090 |
| **Average** | | **~0.48** | **~0.128** |

---

## Pre-Submission Validation

Run the validation script before submitting to catch all disqualifying issues:

```bash
# Make executable
chmod +x validate-submission.sh

# Run against your deployed HF Space
./validate-submission.sh https://YOUR-SPACE.hf.space .
```

The script checks:
1. **HF Space is live** — pings `POST /reset`, expects HTTP 200
2. **Docker build succeeds** — runs `docker build` locally
3. **openenv validate passes** — validates spec compliance

All 3 must pass or your submission will be disqualified.

---

## Python Client Usage

```python
from client import RAGPipelineDoctorEnv
from models import RAGAction

with RAGPipelineDoctorEnv(base_url="https://YOUR_SPACE.hf.space").sync() as env:
    # Start episode
    result = env.reset()
    obs = result.observation
    
    print(f"Task: {obs.task_id} ({obs.difficulty})")
    print(f"Symptoms: {obs.failure_symptoms[0]}")
    
    # Submit a fix
    action = RAGAction(
        diagnosis="The chunk_size of 50 tokens is too small, causing truncation",
        config_patch={"chunk_size": 512, "chunk_overlap": 50},
        confidence=0.9,
    )
    result = env.step(action)
    
    print(f"Reward: {result.reward}")
    print(f"Solved: {result.observation.solved}")
    print(f"Feedback: {result.observation.last_action_feedback}")
    
    # Check state
    state = env.state()
    print(f"Fix score: {state.fix_score}")
```

---

## Using with TRL / GRPO Training

```python
from trl import GRPOConfig, GRPOTrainer
from client import RAGPipelineDoctorEnv
from models import RAGAction

ENV_URL = "https://YOUR_SPACE.hf.space"
env = RAGPipelineDoctorEnv(base_url=ENV_URL)

SYSTEM_PROMPT = """You are an ML engineer diagnosing broken RAG pipelines.
Given a pipeline config and failure symptoms, respond ONLY with JSON:
{"diagnosis": "...", "config_patch": {...}, "confidence": 0.0}"""

def rollout_func(prompts, trainer):
    # ... see inference.py for full implementation
    pass

grpo_config = GRPOConfig(
    use_vllm=True,
    output_dir="outputs/rag-doctor",
    num_train_epochs=1,
    learning_rate=2e-6,
    num_generations=8,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[...],
    train_dataset=dataset,
    args=grpo_config,
    rollout_func=rollout_func,
)
trainer.train()
```

---

## Project Structure

```
rag_pipeline_doctor/
├── openenv.yaml              # OpenEnv spec (spec_version: 1)
├── models.py                 # Typed Pydantic models (Action / Observation / State)
├── tasks.py                  # 3 tasks with broken configs + graders
├── client.py                 # HTTP client (OpenEnv HTTPEnvClient pattern)
├── inference.py              # Baseline inference script — mandatory [START]/[STEP]/[END] format
├── validate-submission.sh    # Pre-submission validation script
├── requirements.txt
├── Dockerfile
├── server/
│   ├── app.py                # FastAPI server — /reset /step /state /health
│   └── rag_environment.py    # Core environment logic
└── tests/
    └── test_env.py           # Smoke tests for all 3 graders
```

---

## Design Decisions

**Why RAG pipelines?** RAG failures are real, expensive, and common in production. Every ML team has hit at least one of these bugs. An agent that can autonomously triage them has immediate practical value.

**Why decomposed rewards?** A single pass/fail reward would give the agent no signal on *why* it failed. Splitting into diagnosis + fix + faithfulness lets the model learn intermediate reasoning skills, not just correct outputs.

**Why 3 retrieval samples per task?** One sample is easy to game. Three samples with consistent failure patterns force the agent to reason about the systemic cause rather than pattern-matching on a single query.

**Why is Task 3 so hard?** Real production incidents almost always have multiple interacting causes. Training on a multi-bug hard task teaches the agent to not stop investigating after finding the first issue.

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | Yes (inference.py) |
| `MODEL_NAME` | Model identifier | Yes (inference.py) |
| `HF_TOKEN` | HuggingFace / API key | Yes (inference.py) |
| `ENV_URL` | RAG Pipeline Doctor Space URL | Yes (inference.py) |

---

## Submission Info

- **HF Space:** `https://huggingface.co/spaces/YOUR_USERNAME/rag-pipeline-doctor`
- **OpenEnv Spec:** `openenv.yaml` (spec_version: 1)
- **Tasks:** 3 (easy / medium / hard), scores 0.0–1.0
- **Baseline:** `inference.py` — completes in < 20 min on 2 vCPU / 8 GB RAM
- **Hackathon:** OpenEnv Hackathon, Meta × PyTorch × Scaler, April 2026
