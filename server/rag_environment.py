"""
RAG Pipeline Doctor — Environment
Implements reset() / step() / state() per the OpenEnv spec.
"""

import uuid
import random
from typing import Optional

from models import (
    RAGAction, RAGObservation, RAGState, RAGStepResult, RetrievalSample
)
from tasks import TASKS, TASK_IDS


MAX_STEPS = 5   # Agent gets 5 attempts to fix the pipeline per episode


class RAGPipelineDoctorEnvironment:
    """
    OpenEnv-compatible environment.
    Each episode:
      1. reset() — sample a random task, expose broken config + symptoms
      2. step(action) — agent submits diagnosis + config_patch, receives reward + feedback
      3. state() — episode metadata (scores, steps, done flag)
    """

    def __init__(self):
        self._state: Optional[RAGState] = None
        self._current_task_id: Optional[str] = None
        self._current_task_data: Optional[dict] = None
        self._current_grader = None
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> RAGStepResult:
        """Start a new episode. Optionally pin a specific task_id for eval."""
        tid = task_id if task_id in TASKS else random.choice(TASK_IDS)
        task_data, grader = TASKS[tid]

        self._current_task_id = tid
        self._current_task_data = task_data
        self._current_grader = grader
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False

        episode_id = str(uuid.uuid4())[:8]
        self._state = RAGState(
            episode_id=episode_id,
            task_id=tid,
            difficulty=task_data["difficulty"],
        )

        obs = self._make_observation(last_feedback=None)
        return RAGStepResult(observation=obs, reward=0.0, done=False, info={"episode_id": episode_id})

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: RAGAction) -> RAGStepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        task_data = self._current_task_data

        # --- Grade the action ---
        diagnosis_score, fix_score, faithfulness_score = self._current_grader(
            action.config_patch, action.diagnosis
        )

        # --- Composite reward ---
        reward = (
            0.35 * diagnosis_score +
            0.45 * fix_score +
            0.20 * faithfulness_score
        )
        # Bonus for fixing on first attempt
        if self._step_count == 1 and fix_score >= 0.9:
            reward = min(1.0, reward + 0.15)
        # Penalty for low-confidence wrong answers
        if action.confidence > 0.8 and fix_score < 0.3:
            reward = max(0.0, reward - 0.05)

        self._cumulative_reward += reward

        # Update state scores
        self._state.total_steps = self._step_count
        self._state.cumulative_reward = round(self._cumulative_reward, 4)
        self._state.diagnosis_score = round(diagnosis_score, 4)
        self._state.fix_score = round(fix_score, 4)
        self._state.faithfulness_score = round(faithfulness_score, 4)

        # Episode ends when pipeline is fixed or max steps reached
        solved = fix_score >= 0.85
        self._done = solved or self._step_count >= MAX_STEPS
        self._state.done = self._done

        # Build feedback for next observation
        feedback = self._build_feedback(action, diagnosis_score, fix_score, faithfulness_score, solved)

        obs = self._make_observation(last_feedback=feedback, solved=solved)
        return RAGStepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info={
                "diagnosis_score": round(diagnosis_score, 4),
                "fix_score": round(fix_score, 4),
                "faithfulness_score": round(faithfulness_score, 4),
                "steps_remaining": MAX_STEPS - self._step_count,
            }
        )

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> RAGState:
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self, last_feedback: Optional[str], solved: bool = False) -> RAGObservation:
        task_data = self._current_task_data
        return RAGObservation(
            task_id=self._current_task_id,
            difficulty=task_data["difficulty"],
            pipeline_config=dict(task_data["broken_config"]),
            failure_symptoms=list(task_data["failure_symptoms"]),
            retrieval_samples=[
                RetrievalSample(**s.model_dump()) for s in task_data["retrieval_samples"]
            ],
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            last_action_feedback=last_feedback,
            solved=solved,
        )

    def _build_feedback(
        self,
        action: RAGAction,
        diagnosis_score: float,
        fix_score: float,
        faithfulness_score: float,
        solved: bool,
    ) -> str:
        lines = [f"=== Step {self._step_count} Feedback ==="]
        lines.append(f"Diagnosis quality : {diagnosis_score:.2f}/1.00")
        lines.append(f"Fix correctness   : {fix_score:.2f}/1.00")
        lines.append(f"Faithfulness      : {faithfulness_score:.2f}/1.00")

        if solved:
            lines.append("Pipeline fixed! Episode complete.")
        elif fix_score < 0.3:
            lines.append("Fix did not resolve the issue. Review the symptoms and try again.")
        elif fix_score < 0.7:
            lines.append("Partial fix. Some issues remain — look deeper at the config.")
        else:
            lines.append("Almost there. One or more parameters still need adjustment.")

        if diagnosis_score < 0.4:
            lines.append("Hint: Your diagnosis didn't identify the core failure mode. "
                         "Focus on what the symptoms have in common.")
        return "\n".join(lines)
