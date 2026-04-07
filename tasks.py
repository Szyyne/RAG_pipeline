"""
RAG Pipeline Doctor — Task Bank
Three tasks: easy → medium → hard.
Each task has:
  - A broken pipeline config
  - Observable symptoms
  - Sample query/retrieval/answer triplets
  - A grader that scores the agent's fix (0.0–1.0)
"""

from typing import Any, Dict, List, Tuple
from models import RetrievalSample


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# TASK 1 — EASY: Chunk Size Too Small
# ---------------------------------------------------------------------------

TASK_CHUNKING_ERROR = {
    "task_id": "chunking_error_001",
    "difficulty": "easy",
    "broken_config": {
        "chunk_size": 50,          # BUG: way too small; should be ~512
        "chunk_overlap": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retriever_top_k": 3,
        "llm_model": "gpt-3.5-turbo",
        "similarity_metric": "cosine",
    },
    "correct_config": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retriever_top_k": 3,
        "llm_model": "gpt-3.5-turbo",
        "similarity_metric": "cosine",
    },
    "failure_symptoms": [
        "Answers are incomplete and cut off mid-sentence",
        "Retrieved chunks contain only 5-10 words each",
        "Context window is filled with many tiny, low-information chunks",
        "Answer faithfulness score: 0.21 (expected > 0.75)",
    ],
    "retrieval_samples": [
        RetrievalSample(
            query="What are the key benefits of transformer architectures?",
            retrieved_chunks=[
                "Transformers use self-att",
                "attention allows parallel",
                "unlike RNNs, transform",
            ],
            expected_answer="Transformers use self-attention to process sequences in parallel, enabling better long-range dependencies and faster training than RNNs.",
            actual_answer="Transformers use self-att... [truncated context]",
            retrieval_score=0.18,
        ),
        RetrievalSample(
            query="How does RLHF improve language model alignment?",
            retrieved_chunks=[
                "RLHF trains a reward mo",
                "human feedback is used",
                "the policy is updated",
            ],
            expected_answer="RLHF uses human feedback to train a reward model, which then guides policy updates to produce more aligned outputs.",
            actual_answer="RLHF trains a reward mo...",
            retrieval_score=0.15,
        ),
        RetrievalSample(
            query="What is the difference between RAG and fine-tuning?",
            retrieved_chunks=[
                "RAG retrieves external",
                "fine-tuning modifies wei",
                "RAG is useful when kno",
            ],
            expected_answer="RAG retrieves external knowledge at inference time while fine-tuning bakes knowledge into model weights.",
            actual_answer="RAG retrieves external...",
            retrieval_score=0.12,
        ),
    ],
}


def grade_chunking_error(config_patch: Dict[str, Any], diagnosis: str) -> Tuple[float, float, float]:
    """Returns (diagnosis_score, fix_score, faithfulness_score)."""
    # Diagnosis score: did the agent identify chunk_size as the problem?
    diag_keywords = ["chunk", "size", "truncat", "small", "50", "overlap"]
    diag_hits = sum(1 for kw in diag_keywords if kw.lower() in diagnosis.lower())
    diagnosis_score = _clamp(diag_hits / 3.0)

    # Fix score: did the agent patch chunk_size to something reasonable (256–1024)?
    patched_size = config_patch.get("chunk_size", 0)
    if 256 <= patched_size <= 1024:
        fix_score = 1.0
    elif 100 <= patched_size < 256:
        fix_score = 0.5
    else:
        fix_score = 0.0

    # Overlap bonus
    patched_overlap = config_patch.get("chunk_overlap", -1)
    if patched_overlap >= 20:
        fix_score = min(1.0, fix_score + 0.1)

    faithfulness_score = fix_score * 0.9  # proxy: if config is fixed, answers improve

    return diagnosis_score, fix_score, faithfulness_score


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM: Embedding Model Mismatch
# ---------------------------------------------------------------------------

TASK_EMBEDDING_MISMATCH = {
    "task_id": "embedding_mismatch_001",
    "difficulty": "medium",
    "broken_config": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "index_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",   # BUG: indexed with MiniLM
        "query_embedding_model": "BAAI/bge-large-en-v1.5",                    # BUG: querying with BGE
        "retriever_top_k": 5,
        "llm_model": "gpt-3.5-turbo",
        "similarity_metric": "cosine",
    },
    "correct_config": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "index_embedding_model": "BAAI/bge-large-en-v1.5",
        "query_embedding_model": "BAAI/bge-large-en-v1.5",
        "retriever_top_k": 5,
        "llm_model": "gpt-3.5-turbo",
        "similarity_metric": "cosine",
    },
    "failure_symptoms": [
        "Retriever returns documents that seem topically unrelated to the query",
        "Cosine similarity scores between query and retrieved chunks are anomalously low (0.05–0.15)",
        "Increasing top_k did not improve answer quality",
        "Re-indexing documents did not resolve the issue",
        "Answer faithfulness score: 0.19 (expected > 0.75)",
    ],
    "retrieval_samples": [
        RetrievalSample(
            query="Explain the vanishing gradient problem in deep networks",
            retrieved_chunks=[
                "Photosynthesis is a process used by plants...",
                "The French Revolution began in 1789...",
                "Quantum entanglement describes correlations...",
            ],
            expected_answer="In deep networks, gradients shrink exponentially during backpropagation, making early layers learn very slowly.",
            actual_answer="I cannot find relevant information about gradient problems in the provided context.",
            retrieval_score=0.06,
        ),
        RetrievalSample(
            query="What is the role of the key-query-value mechanism in attention?",
            retrieved_chunks=[
                "The mitochondria is the powerhouse of the cell...",
                "World War II ended in 1945...",
                "Python is a high-level programming language...",
            ],
            expected_answer="Keys and queries compute attention weights via dot-product, which determine how much each value contributes to the output.",
            actual_answer="The provided context does not contain information about attention mechanisms.",
            retrieval_score=0.04,
        ),
        RetrievalSample(
            query="How does dropout regularization prevent overfitting?",
            retrieved_chunks=[
                "The Amazon rainforest covers 5.5 million km²...",
                "Marie Curie won two Nobel prizes...",
                "The speed of light is 299,792 km/s...",
            ],
            expected_answer="Dropout randomly deactivates neurons during training, forcing the network to learn redundant representations and reducing co-adaptation.",
            actual_answer="I don't have relevant context about dropout regularization.",
            retrieval_score=0.03,
        ),
    ],
}


def grade_embedding_mismatch(config_patch: Dict[str, Any], diagnosis: str) -> Tuple[float, float, float]:
    """Returns (diagnosis_score, fix_score, faithfulness_score)."""
    diag_keywords = ["embedding", "mismatch", "index", "query", "different model", "inconsistent", "bge", "minilm"]
    diag_hits = sum(1 for kw in diag_keywords if kw.lower() in diagnosis.lower())
    diagnosis_score = _clamp(diag_hits / 3.0)

    index_model = config_patch.get("index_embedding_model", "")
    query_model = config_patch.get("query_embedding_model", "")

    if index_model and query_model and index_model == query_model:
        fix_score = 1.0
    elif index_model == query_model == "":
        fix_score = 0.0
    elif index_model or query_model:
        # Partial: agent patched one but not both
        fix_score = 0.4
    else:
        fix_score = 0.0

    faithfulness_score = fix_score * 0.95

    return diagnosis_score, fix_score, faithfulness_score


# ---------------------------------------------------------------------------
# TASK 3 — HARD: Silent Semantic Drift (wrong similarity metric + stale index)
# ---------------------------------------------------------------------------

TASK_HALLUCINATION_RETRIEVAL = {
    "task_id": "hallucination_retrieval_001",
    "difficulty": "hard",
    "broken_config": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "retriever_top_k": 3,
        "llm_model": "gpt-3.5-turbo",
        "similarity_metric": "dot_product",      # BUG 1: should be cosine for normalised BGE embeddings
        "index_version": "v1.2",                 # BUG 2: stale index (docs updated but not re-indexed)
        "reranker_enabled": False,               # BUG 3: reranker disabled, so noisy top-k passes through
        "context_window_strategy": "first_k",   # BUG 4: should be "most_relevant" not just first k chunks
    },
    "correct_config": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "retriever_top_k": 5,
        "llm_model": "gpt-3.5-turbo",
        "similarity_metric": "cosine",
        "index_version": "latest",
        "reranker_enabled": True,
        "context_window_strategy": "most_relevant",
    },
    "failure_symptoms": [
        "LLM produces confident-sounding but factually wrong answers",
        "Retrieved documents are from the correct topic domain but contain outdated information",
        "Answers pass a surface-level relevance check but fail factual verification",
        "Faithfulness score is deceptively moderate (0.45) — answers sound plausible",
        "Bug is subtle: retrieval appears to be working but is silently misleading the LLM",
        "Answer faithfulness score: 0.45 (expected > 0.80) — particularly dangerous failure mode",
    ],
    "retrieval_samples": [
        RetrievalSample(
            query="What is the current SOTA on MMLU benchmark?",
            retrieved_chunks=[
                "[v1.2 index — outdated] GPT-4 achieves 86.4% on MMLU as of March 2023.",
                "[v1.2 index — outdated] PaLM 2 scores 85.1% on MMLU benchmark.",
                "[v1.2 index — outdated] Claude 2 achieves 78.5% on MMLU.",
            ],
            expected_answer="As of 2024, models like GPT-4o, Gemini Ultra, and Claude 3 Opus exceed 88–90% on MMLU.",
            actual_answer="GPT-4 achieves 86.4% on MMLU, which is the current state of the art.",
            retrieval_score=0.41,
        ),
        RetrievalSample(
            query="What are the best practices for RAG system evaluation?",
            retrieved_chunks=[
                "[v1.2 index — outdated] Use BLEU and ROUGE to evaluate RAG outputs.",
                "[v1.2 index — outdated] Human evaluation is the gold standard for RAG.",
                "[v1.2 index — outdated] Retrieval precision@k is sufficient for RAG evaluation.",
            ],
            expected_answer="Modern RAG evaluation uses frameworks like RAGAS, measuring retrieval recall, answer faithfulness, and context relevance separately.",
            actual_answer="Use BLEU, ROUGE, and precision@k metrics. Human evaluation remains the gold standard.",
            retrieval_score=0.38,
        ),
        RetrievalSample(
            query="How should I handle long documents in a RAG pipeline?",
            retrieved_chunks=[
                "[v1.2 index — outdated] Split documents into 512-token chunks without overlap.",
                "[v1.2 index — outdated] Use a sliding window of 256 tokens.",
                "[v1.2 index — outdated] Summarise long documents before indexing.",
            ],
            expected_answer="Modern approaches include hierarchical indexing, late chunking, contextual retrieval (Anthropic), and proposition indexing for better semantic preservation.",
            actual_answer="Split documents into 512-token chunks or use a 256-token sliding window. Summarising is also an option.",
            retrieval_score=0.43,
        ),
    ],
}


def grade_hallucination_retrieval(config_patch: Dict[str, Any], diagnosis: str) -> Tuple[float, float, float]:
    """Returns (diagnosis_score, fix_score, faithfulness_score). Harder partial scoring."""
    diag_keywords = [
        "stale", "outdated", "index", "rerank", "dot product", "cosine",
        "similarity metric", "hallucin", "context window", "most_relevant", "v1.2"
    ]
    diag_hits = sum(1 for kw in diag_keywords if kw.lower() in diagnosis.lower())
    # Hard task: need to identify multiple bugs
    diagnosis_score = _clamp(diag_hits / 4.0)

    fix_points = 0
    total_points = 4

    if config_patch.get("similarity_metric") == "cosine":
        fix_points += 1
    if config_patch.get("index_version") in ("latest", "v2", "current"):
        fix_points += 1
    if config_patch.get("reranker_enabled") is True:
        fix_points += 1
    if config_patch.get("context_window_strategy") == "most_relevant":
        fix_points += 1

    fix_score = _clamp(fix_points / total_points)
    faithfulness_score = _clamp(fix_score * 0.9 + diagnosis_score * 0.1)

    return diagnosis_score, fix_score, faithfulness_score


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "chunking_error_001": (TASK_CHUNKING_ERROR, grade_chunking_error),
    "embedding_mismatch_001": (TASK_EMBEDDING_MISMATCH, grade_embedding_mismatch),
    "hallucination_retrieval_001": (TASK_HALLUCINATION_RETRIEVAL, grade_hallucination_retrieval),
}

TASK_IDS = list(TASKS.keys())
