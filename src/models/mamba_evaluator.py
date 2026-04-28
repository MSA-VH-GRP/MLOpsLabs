"""
Ranking evaluation metrics for Mamba4Rec.

Metrics: Hit@K, NDCG@K, MRR@K
Computed over a candidate set of (target + negatives) where target is at index 0.

Note: this file is separate from evaluator.py (which has the sklearn accuracy helpers)
to avoid any naming collision.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


def compute_metrics(
    scores: torch.Tensor,
    k: int = 10,
    target_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Hit@K, NDCG@K, and MRR@K for one batch.

    Args:
        scores:     (batch, num_candidates) — target item is at target_idx
        k:          Top-K cut-off
        target_idx: Column index of the target item in the candidate list (default 0)

    Returns:
        (hits, ndcgs, mrrs) each of shape (batch,)
    """
    _, indices = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)

    # Hit@K
    target_in_topk = (indices == target_idx).any(dim=1)
    hits = target_in_topk.float()

    # Rank of the target item (1-indexed)
    target_score = scores[:, target_idx : target_idx + 1]
    rank = (scores > target_score).sum(dim=1) + 1

    # NDCG@K = 1/log2(rank+1) if rank <= k, else 0
    ndcgs = torch.where(
        rank <= k,
        1.0 / torch.log2(rank.float() + 1),
        torch.zeros_like(rank.float()),
    )

    # MRR@K = 1/rank if rank <= k, else 0
    mrrs = torch.where(
        rank <= k,
        1.0 / rank.float(),
        torch.zeros_like(rank.float()),
    )

    return hits, ndcgs, mrrs


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Full model evaluation over a candidate-based test loader.

    Args:
        model:       Mamba4Rec (or any model with predict_scores)
        test_loader: DataLoader yielding EvalDataset batches
        device:      Compute device
        ks:          List of K cut-offs

    Returns:
        Dict with keys Hit@K, NDCG@K, MRR@K for each K
    """
    model.eval()

    all_hits  = {k: [] for k in ks}
    all_ndcgs = {k: [] for k in ks}
    all_mrrs  = {k: [] for k in ks}

    for batch in test_loader:
        item_seq    = batch["item_seq"].to(device)
        genre_seq   = batch["genre_seq"].to(device)
        time_seq    = batch["time_seq"].to(device)
        delta_seq   = batch["delta_seq"].to(device) if "delta_seq" in batch else None
        age_idx     = batch["age_idx"].to(device)
        gender_idx  = batch["gender_idx"].to(device)
        occupation  = batch["occupation"].to(device)
        candidates  = batch["candidates"].to(device)

        scores = model.predict_scores(
            item_seq, genre_seq, time_seq,
            age_idx, gender_idx, occupation,
            candidate_items=candidates,
            delta_seq=delta_seq,
        )

        for k in ks:
            hits, ndcgs, mrrs = compute_metrics(scores, k=k)
            all_hits[k].append(hits)
            all_ndcgs[k].append(ndcgs)
            all_mrrs[k].append(mrrs)

    results: Dict[str, float] = {}
    for k in ks:
        results[f"Hit@{k}"]  = torch.cat(all_hits[k]).mean().item()
        results[f"NDCG@{k}"] = torch.cat(all_ndcgs[k]).mean().item()
        results[f"MRR@{k}"]  = torch.cat(all_mrrs[k]).mean().item()

    return results
