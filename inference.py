"""
RAG Pipeline Doctor — Baseline Inference Script
================================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        URL of the deployed RAG Pipeline Doctor HF Space.

STDOUT FORMAT (mandatory per hackathon rules):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct 
    HF_TOKEN=hf_... 
    ENV_URL=https://YOUR-SPACE.hf.space 
    python inference.py
"""

import json
import os
import sys
import textwrap
import time
from typing import List, Optional

from openai import OpenAI

# Load .env file if it exists (for local convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables as-is

from client import RAGPipelineDoctorEnv
from models import RAGAction

# ---------------------------------------------------------------------------
# Config — read from environment variables (mandatory per rules)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
BENCHMARK    = "rag_pipeline_doctor"

MAX_STEPS               = 5
TEMPERATURE             = 0.7  # Increased for more variety
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.5   # normalized score threshold for success

TASK_IDS = [
    "chunking_error_001",
    "embedding_mismatch_001",
    "hallucination_retrieval_001",
]


# ---------------------------------------------------------------------------
# Mandatory stdout logging helpers — [START] / [STEP] / [END]
# Deviation from this format causes incorrect evaluation scoring.
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # action must be on a single line
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior ML engineer specialising in RAG (Retrieval-Augmented Generation) systems.

    You will receive a broken RAG pipeline configuration together with:
    - Observable failure symptoms
    - Sample query -> retrieved chunks -> actual answer triplets (vs expected answers)
    - Feedback from your previous fix attempt (if any)

    Your job:
    1. Diagnose the root cause of the pipeline failure
    2. Propose a config patch (key-value dict) that fixes the pipeline

    IMPORTANT: Be thorough and systematic. Check ALL parameters that could be wrong.
    Don't just fix the obvious issue - look for multiple problems that could cause the symptoms.

    Respond ONLY with valid JSON on one line — no markdown, no explanation outside the JSON:
    {"diagnosis": "Concise root-cause explanation", "config_patch": {"key": "value"}, "confidence": 0.9}
""").strip()


def build_user_prompt(obs_dict: dict) -> str:
    config_str = json.dumps(obs_dict["pipeline_config"], indent=2)
    symptoms   = "\n".join(f"  - {s}" for s in obs_dict["failure_symptoms"])

    samples_lines = []
    for i, s in enumerate(obs_dict["retrieval_samples"], 1):
        samples_lines.append(f"\n  Sample {i}:")
        samples_lines.append(f"    Query    : {s['query']}")
        samples_lines.append(f"    Retrieved: {s['retrieved_chunks']}")
        samples_lines.append(f"    Expected : {s['expected_answer']}")
        samples_lines.append(f"    Actual   : {s['actual_answer']}")
        samples_lines.append(f"    Score    : {s['retrieval_score']:.2f}")

    feedback = obs_dict.get("last_action_feedback") or "None (first attempt)"

    return (
        f"BROKEN PIPELINE CONFIG:\n{config_str}\n\n"
        f"FAILURE SYMPTOMS:\n{symptoms}\n\n"
        f"RETRIEVAL SAMPLES:{''.join(samples_lines)}\n\n"
        f"PREVIOUS FEEDBACK: {feedback}\n\n"
        f"Task difficulty : {obs_dict['difficulty']}\n"
        f"Attempt         : {obs_dict['step_number'] + 1} / {obs_dict['max_steps']}\n\n"
        f"Diagnose and fix this pipeline. Respond only with JSON."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, obs_dict: dict):
    """Call the LLM, return (RAGAction, compact_action_str_for_logging)."""
    user_prompt = build_user_prompt(obs_dict)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM/parse error: {exc}", flush=True)
        parsed = {"diagnosis": f"parse_error", "config_patch": {}, "confidence": 0.1}

    action = RAGAction(
        diagnosis    = str(parsed.get("diagnosis", "")),
        config_patch = dict(parsed.get("config_patch", {})),
        confidence   = float(parsed.get("confidence", 0.5)),
    )
    # Compact single-line string for [STEP] log
    action_str = json.dumps({"patch": action.config_patch}, separators=(",", ":"))
    return action, action_str


# ---------------------------------------------------------------------------
# Single episode — emits [START] / [STEP]* / [END]
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, env: RAGPipelineDoctorEnv, task_id: str) -> dict:
    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0
    done:        bool        = False
    last_obs                 = None
    last_state               = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = env.reset(task_id=task_id)
        last_obs   = result.observation

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action, action_str = get_model_action(client, last_obs.model_dump())

            step_result = env.step(action)
            last_obs    = step_result.observation
            reward      = step_result.reward or 0.0
            done        = step_result.done
            error       = step_result.info.get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score   = sum(rewards) / MAX_STEPS          # normalise to [0, 1]
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        last_state = env.state()

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":         task_id,
        "steps":           steps_taken,
        "score":           round(score, 4),
        "success":         success,
        "rewards":         rewards,
        "fix_score":       round(last_state.fix_score, 4)        if last_state else 0.0,
        "diagnosis_score": round(last_state.diagnosis_score, 4)  if last_state else 0.0,
        "solved":          last_obs.solved                        if last_obs   else False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()

    client = OpenAI(
        api_key  = HF_TOKEN,
        base_url = API_BASE_URL,
    )

    all_results = []

    with RAGPipelineDoctorEnv(base_url=ENV_URL) as env:
        try:
            env.health()
        except Exception as exc:
            print(f"[DEBUG] Health check warning: {exc}", flush=True)

        for task_id in TASK_IDS:
            result = run_episode(client, env, task_id=task_id)
            all_results.append(result)

    elapsed = time.time() - start_time

    # Human-readable summary (does not affect automated scoring)
    print("\n" + "=" * 62, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 62, flush=True)
    print(f"{'Task':<38} {'Score':>6} {'Fix':>6} {'Solved':>7}", flush=True)
    print("-" * 62, flush=True)
    for r in all_results:
        icon = "YES" if r["solved"] else "NO"
        print(f"{r['task_id']:<38} {r['score']:>6.3f} {r['fix_score']:>6.2f} {icon:>7}", flush=True)
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print("-" * 62, flush=True)
    print(f"{'AVERAGE':<38} {avg_score:>6.3f}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s  |  Model: {MODEL_NAME}", flush=True)

    if avg_score < 0.01:
        print("\n[ERROR] All scores near zero — check ENV_URL and API credentials.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
