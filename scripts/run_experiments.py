"""
Experiment runner for Mamba4Rec user-fusion ablation + fine-tuning study.

Calls POST /train for each config, fetches results from MLflow,
and prints a formatted comparison table when all runs complete.

Usage (run inside the api container):
    docker exec mlops-api bash -c "cd /app && python scripts/run_experiments.py [--suite SUITE]"

Suites:
    fusion      — original 3 fusion modes (broadcast / film / head)
    finetune    — 6 fine-tuning experiments on Mamba+Head
    longseq     — long sequence hypothesis: GRU vs Mamba at max_seq_len=200
    all         — run all suites
"""

import argparse
import json
import time
import requests
from datetime import datetime

# ── URLs (inside Docker network) ──────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
MLFLOW_BASE = "http://mlflow:5000"
EXPERIMENT_NAME = "mamba4rec-user-fusion"

# ─── Shared base hyperparams ──────────────────────────────────────────────────
BASE_HP = {
    "d_model":          64,
    "d_state":          16,
    "n_layers":          3,
    "d_conv":            4,
    "expand":            2,
    "dropout":          0.3,
    "max_seq_len":       50,
    "batch_size":       128,
    "learning_rate":   2e-4,
    "weight_decay":    1e-3,
    "epochs":            50,
    "patience":          10,
    "min_seq_len":        5,
    "max_train_per_user": 20,
    "label_smoothing":  0.1,
    "warmup_epochs":     10,
    "lr_scheduler":  "plateau",
    "plateau_patience":   3,
    "plateau_factor":   0.5,
    "eta_min":         1e-5,
    "use_sum_token":  False,
    "use_tupe":       False,
    "use_time_interval": True,
    "num_time_interval_bins": 256,
}

# ─── Suite 1: Original fusion ablation ────────────────────────────────────────
FUSION_EXPERIMENTS = [
    {
        "name": "TI+Broadcast",
        "hp": {**BASE_HP, "user_fusion_mode": "broadcast"},
    },
    {
        "name": "TI+FiLM",
        "hp": {**BASE_HP, "user_fusion_mode": "film"},
    },
    {
        "name": "TI+Head",
        "hp": {**BASE_HP, "user_fusion_mode": "head"},
    },
]

# ─── Suite 2: Fine-tuning experiments ─────────────────────────────────────────
# All build on top of Mamba+Head (best Mamba config from ablation).

# Priority 1: Gated head + d_state=32 (architectural change, high impact)
_HP_GATED = {
    **BASE_HP,
    "user_fusion_mode": "gated_head",
    "d_state": 32,               # larger Mamba memory capacity
}

# Priority 2: Lower LR + less dropout (HP tuning for head modes)
_HP_LR = {
    **BASE_HP,
    "user_fusion_mode": "gated_head",
    "d_state": 32,
    "learning_rate": 1e-4,       # lower LR — head projection learns carefully
    "dropout": 0.2,              # less regularization for smaller head proj
    "plateau_patience": 5,       # more patient scheduler
}

# Priority 3: Two-stage training (freeze backbone first)
_HP_TWO_STAGE = {
    **_HP_LR,
    "two_stage_training": True,
    "stage1_epochs": 20,         # 20 epochs backbone-only, then unfreeze head
    "warmup_epochs": 15,         # longer warmup for stable backbone convergence
}

# Priority 4: Normed head (LayerNorm after head fusion)
_HP_NORMED = {
    **_HP_LR,
    "user_fusion_mode": "normed_head",
}

# Priority 5: Hybrid fusion (light broadcast alpha + gated head)
_HP_HYBRID = {
    **_HP_LR,
    "user_fusion_mode": "hybrid",
}

# Priority 6: Two-stage + normed head (combining best techniques)
_HP_TWO_STAGE_NORMED = {
    **_HP_LR,
    "user_fusion_mode": "normed_head",
    "two_stage_training": True,
    "stage1_epochs": 20,
    "warmup_epochs": 15,
}

# ─── Suite 3: Long sequence hypothesis ───────────────────────────────────────
# Test whether Mamba outperforms GRU when context is significantly longer.
# Hypothesis: Mamba's selective scan advantage appears at L > 100.
# Changes vs BASE_HP:
#   max_seq_len:       50  →  200   (4× longer context window)
#   max_train_per_user: 20  →  100   (4× more training samples per user)
#   d_state:            16  →  64    (Mamba needs larger state for long sequences)
#   expand:              2  →   4    (wider SSM dimension)
#   batch_size:        128  →   64   (larger sequences → more GPU memory per sample)
#   epochs:             50  →   30   (each epoch is slower; use early stopping)
_LONG_HP = {
    **BASE_HP,
    "max_seq_len":        200,
    "max_train_per_user": 100,
    "d_state":             64,
    "expand":               4,
    "batch_size":          64,
    "epochs":              30,
    "patience":            8,
    "warmup_epochs":       5,
}

LONGSEQ_EXPERIMENTS = [
    {
        "name": "GRU+Broadcast+L200",
        "hp": {**_LONG_HP, "user_fusion_mode": "broadcast", "force_gru": True},
    },
    {
        "name": "Mamba+Broadcast+L200",
        "hp": {**_LONG_HP, "user_fusion_mode": "broadcast", "force_gru": False},
    },
    {
        "name": "Mamba+GatedHead+L200",
        "hp": {**_LONG_HP, "user_fusion_mode": "gated_head", "d_state": 64, "force_gru": False},
    },
]

FINETUNE_EXPERIMENTS = [
    {"name": "Gated+d32",          "hp": _HP_GATED},
    {"name": "Gated+d32+LR1e-4",   "hp": _HP_LR},
    {"name": "TwoStage+Gated",      "hp": _HP_TWO_STAGE},
    {"name": "NormedHead+LR1e-4",  "hp": _HP_NORMED},
    {"name": "Hybrid+LR1e-4",      "hp": _HP_HYBRID},
    {"name": "TwoStage+Normed",    "hp": _HP_TWO_STAGE_NORMED},
]


def post_train(exp_name: str, hp: dict) -> dict:
    payload = {
        "experiment_name": exp_name,
        "model_type": "mamba4rec",
        "hyperparams": hp,
    }
    resp = requests.post(f"{API_BASE}/train", json=payload, timeout=7200)
    resp.raise_for_status()
    return resp.json()


def get_mlflow_run(run_id: str) -> tuple[dict, dict]:
    url = f"{MLFLOW_BASE}/api/2.0/mlflow/runs/get?run_id={run_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()["run"]["data"]
    metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
    params  = {p["key"]: p["value"] for p in data.get("params",  [])}
    return metrics, params


def format_table(results: list[dict]) -> str:
    header = "| Experiment            | Backbone | Hit@10 | NDCG@10 | MRR@10 | Hit@5  | NDCG@5  | Time  |"
    sep    = "|:----------------------|:---------|-------:|--------:|-------:|-------:|--------:|------:|"
    rows = [header, sep]
    for r in results:
        m = r.get("metrics", {})
        t = f"{r.get('elapsed', 0)/60:.1f}m"
        rows.append(
            f"| {r['name']:<21} | {r.get('backbone','?'):<8} | "
            f"{m.get('test_hit_at_10',  0):.4f} | "
            f"{m.get('test_ndcg_at_10', 0):.4f}  | "
            f"{m.get('test_mrr_at_10',  0):.4f} | "
            f"{m.get('test_hit_at_5',   0):.4f} | "
            f"{m.get('test_ndcg_at_5',  0):.4f}  | {t:>5} |"
        )
    return "\n".join(rows)


def run_suite(suite_name: str, experiments: list[dict]) -> list[dict]:
    print(f"\n{'='*65}")
    print(f"  Suite: {suite_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    results = []
    for i, exp in enumerate(experiments, 1):
        fusion = exp["hp"].get("user_fusion_mode", "broadcast")
        two_stage = exp["hp"].get("two_stage_training", False)
        force_gru = exp["hp"].get("force_gru", False)
        print(f"[{i}/{len(experiments)}] {exp['name']}")
        print(f"  fusion={fusion}  two_stage={two_stage}  d_state={exp['hp'].get('d_state',16)}  force_gru={force_gru}")
        print(f"  Posting to /train ...", flush=True)

        t0 = time.time()
        try:
            resp = post_train(EXPERIMENT_NAME, exp["hp"])
            elapsed = time.time() - t0
            run_id = resp.get("run_id", "")

            metrics, params = {}, {}
            if run_id:
                try:
                    metrics, params = get_mlflow_run(run_id)
                except Exception as e:
                    print(f"  ⚠ MLflow fetch failed: {e}")

            for k, v in resp.items():
                if k not in ("run_id", "status", "message") and isinstance(v, (int, float)):
                    metrics.setdefault(k, v)

            ndcg = metrics.get("test_ndcg_at_10", 0)
            hit  = metrics.get("test_hit_at_10",  0)
            mrr  = metrics.get("test_mrr_at_10",  0)
            mamba_ok = params.get("mamba_ssm_available", "")
            backbone = "mamba" if str(mamba_ok).lower() == "true" else "gru"

            print(f"  ✓ Done in {elapsed/60:.1f} min")
            print(f"    NDCG@10={ndcg:.4f}  Hit@10={hit:.4f}  MRR@10={mrr:.4f}")
            print(f"    backbone={backbone}  run_id={run_id}")

            results.append({
                "name": exp["name"], "backbone": backbone,
                "run_id": run_id, "metrics": metrics, "elapsed": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ FAILED after {elapsed/60:.1f} min: {e}")
            results.append({
                "name": exp["name"], "backbone": "?",
                "run_id": "", "metrics": {}, "elapsed": elapsed,
            })
        print()

    print(f"\n{'='*65}")
    print(f"  RESULTS — {suite_name}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")
    print(format_table(results))
    print()

    valid = [r for r in results if r["metrics"].get("test_ndcg_at_10", 0) > 0]
    if valid:
        best = max(valid, key=lambda r: r["metrics"]["test_ndcg_at_10"])
        print(f"🏆 Best [{suite_name}]: {best['name']}  NDCG@10={best['metrics']['test_ndcg_at_10']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="finetune",
                        choices=["fusion", "finetune", "longseq", "all"],
                        help="Which experiment suite to run")
    args = parser.parse_args()

    all_results = []

    if args.suite in ("fusion", "all"):
        all_results += run_suite("User Fusion Ablation", FUSION_EXPERIMENTS)

    if args.suite in ("finetune", "all"):
        all_results += run_suite("Fine-Tuning (Mamba+Head)", FINETUNE_EXPERIMENTS)

    if args.suite in ("longseq", "all"):
        all_results += run_suite("Long Sequence (GRU vs Mamba @ L=200)", LONGSEQ_EXPERIMENTS)

    if args.suite == "all" and all_results:
        print(f"\n{'='*65}")
        print("  COMBINED RESULTS (all suites)")
        print(f"{'='*65}\n")
        print(format_table(all_results))
        valid = [r for r in all_results if r["metrics"].get("test_ndcg_at_10", 0) > 0]
        if valid:
            best = max(valid, key=lambda r: r["metrics"]["test_ndcg_at_10"])
            print(f"\n🏆 Overall best: {best['name']}  NDCG@10={best['metrics']['test_ndcg_at_10']:.4f}")

    out_path = "/app/scripts/experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
