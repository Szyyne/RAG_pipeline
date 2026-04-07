"""
RAG Pipeline Doctor — Local Tests
Smoke-tests the environment without any HTTP server.
Run: python test_env.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.rag_environment import RAGPipelineDoctorEnvironment
from models import RAGAction


def test_episode(task_id: str, patch: dict, diagnosis: str, label: str):
    env = RAGPipelineDoctorEnvironment()
    r = env.reset(task_id=task_id)
    print(f"\n[{label}] task={task_id}  diff={r.observation.difficulty}")
    print(f"  Symptoms: {r.observation.failure_symptoms[0]}")

    step = env.step(RAGAction(diagnosis=diagnosis, config_patch=patch, confidence=0.9))
    info = step.info
    print(f"  reward={step.reward:.3f}  fix={info['fix_score']:.2f}  "
          f"diag={info['diagnosis_score']:.2f}  done={step.done}  solved={step.observation.solved}")
    assert 0.0 <= step.reward <= 1.0, "Reward out of bounds"
    return step.reward


if __name__ == "__main__":
    print("=" * 55)
    print("RAG Pipeline Doctor — Environment Smoke Tests")
    print("=" * 55)

    # Task 1 — correct fix
    r1 = test_episode(
        task_id="chunking_error_001",
        patch={"chunk_size": 512, "chunk_overlap": 50},
        diagnosis="The chunk_size of 50 tokens is far too small, causing truncation. Should be ~512 with overlap.",
        label="EASY (correct fix)",
    )
    assert r1 > 0.5, f"Expected reward > 0.5, got {r1}"

    # Task 1 — wrong fix
    r2 = test_episode(
        task_id="chunking_error_001",
        patch={"retriever_top_k": 10},
        diagnosis="The retriever is not returning enough documents.",
        label="EASY (wrong fix)",
    )
    assert r2 < 0.5, f"Expected reward < 0.5, got {r2}"

    # Task 2 — correct fix
    r3 = test_episode(
        task_id="embedding_mismatch_001",
        patch={"index_embedding_model": "BAAI/bge-large-en-v1.5",
               "query_embedding_model": "BAAI/bge-large-en-v1.5"},
        diagnosis="The index was built with MiniLM but queries use BGE — embedding space mismatch.",
        label="MEDIUM (correct fix)",
    )
    assert r3 > 0.5, f"Expected reward > 0.5, got {r3}"

    # Task 3 — full correct fix
    r4 = test_episode(
        task_id="hallucination_retrieval_001",
        patch={
            "similarity_metric": "cosine",
            "index_version": "latest",
            "reranker_enabled": True,
            "context_window_strategy": "most_relevant",
        },
        diagnosis="Stale index (v1.2), wrong similarity metric (dot_product instead of cosine), "
                  "reranker disabled, and first_k context strategy together cause silent hallucinations.",
        label="HARD (correct fix)",
    )
    assert r4 > 0.6, f"Expected reward > 0.6, got {r4}"

    # Task 3 — partial fix
    r5 = test_episode(
        task_id="hallucination_retrieval_001",
        patch={"similarity_metric": "cosine", "index_version": "latest"},
        diagnosis="Stale index and wrong similarity metric.",
        label="HARD (partial fix)",
    )
    assert 0.2 < r5 < 0.9, f"Expected partial reward, got {r5}"

    # State check
    env = RAGPipelineDoctorEnvironment()
    env.reset()
    s = env.state()
    assert s.total_steps == 0
    assert s.done is False
    print(f"\n[STATE] episode_id={s.episode_id}  task={s.task_id}  done={s.done} ✅")

    print("\n" + "=" * 55)
    print("All tests passed ✅")
    print("=" * 55)
