"""Reusable metric helpers for PolyEPO / GRPO advantage estimators."""

import numpy as np
import torch


def success_rate_metrics(scores: torch.Tensor, id2indices: dict) -> dict:
    """Per-prompt success rate: fraction of prompts with any response scoring >= 1."""
    num_prompts_total = 0
    num_prompts_with_success = 0
    for batch_indices in id2indices.values():
        if not batch_indices:
            continue
        num_prompts_total += 1
        prompt_scores = scores[torch.tensor(batch_indices, device=scores.device, dtype=torch.long)]
        if (prompt_scores >= 1).any().item():
            num_prompts_with_success += 1
    success_rate = (num_prompts_with_success / num_prompts_total) if num_prompts_total > 0 else 0.0
    return {
        "training_logs/grpo_num_prompts_with_success": num_prompts_with_success,
        "training_logs/grpo_prompt_success_rate": success_rate,
    }


def cluster_diversity_metrics(
    api_jobs: list[dict],
    all_cluster_assignments: list,
    id2indices: dict,
    raw_scores: torch.Tensor,
) -> dict:
    """Cluster-based diversity metrics from precomputed cluster-judge output.

    Args:
        api_jobs: list of dicts each carrying a 'prompt_id' field; parallel to
            all_cluster_assignments (one entry per prompt sent to the judge).
        all_cluster_assignments: list of cluster-id lists (or empty/None for
            prompts where the judge call failed).
        id2indices: prompt_id -> list of batch indices in the response tensor.
        raw_scores: (bsz,) tensor of per-response raw rewards, used to split
            unique-cluster counts by correct (> 0) vs incorrect (== 0).
    """
    out: dict = {}
    if not api_jobs or not all_cluster_assignments:
        return out

    all_unique_counts: list[int] = []
    correct_unique_counts: list[int] = []
    wrong_unique_counts: list[int] = []
    cluster_100_counts: list[int] = []

    for job, assignments in zip(api_jobs, all_cluster_assignments):
        if not assignments:
            continue
        batch_indices = id2indices[job["prompt_id"]]
        if len(assignments) != len(batch_indices):
            continue
        group_raw_scores = raw_scores[batch_indices]
        all_unique_counts.append(len(set(assignments)))
        cluster_100_counts.append(sum(1 for c in assignments if c == 100))
        correct_mask = (group_raw_scores > 0)
        wrong_mask = (group_raw_scores == 0)
        if correct_mask.any():
            c_indices = correct_mask.nonzero(as_tuple=True)[0]
            c_clusters = [assignments[i.item()] for i in c_indices]
            correct_unique_counts.append(len(set(c_clusters)))
        if wrong_mask.any():
            w_indices = wrong_mask.nonzero(as_tuple=True)[0]
            w_clusters = [assignments[i.item()] for i in w_indices]
            wrong_unique_counts.append(len(set(w_clusters)))

    if all_unique_counts:
        # Divisor leaks `assignments` from the last loop iteration — preserved from original.
        out.update({
            "diversity/grpo_path_diversity": np.mean(all_unique_counts) / (len(assignments) if assignments else 1),
            "clusters/avg_unique_clusters_across_prompts": np.mean(all_unique_counts),
            "clusters/var_unique_clusters_across_prompts": np.var(all_unique_counts) if len(all_unique_counts) > 1 else 0.0,
        })
    if correct_unique_counts:
        out.update({
            "clusters/avg_unique_clusters_for_correct_ans": np.mean(correct_unique_counts),
            "clusters/var_unique_clusters_for_correct_ans": np.var(correct_unique_counts) if len(correct_unique_counts) > 1 else 0.0,
        })
    if wrong_unique_counts:
        out.update({
            "clusters/avg_unique_clusters_for_incorrect_ans": np.mean(wrong_unique_counts),
            "clusters/var_unique_clusters_for_incorrect_ans": np.var(wrong_unique_counts) if len(wrong_unique_counts) > 1 else 0.0,
        })
    if cluster_100_counts:
        out["clusters/avg_cluster_100_generations_per_prompt"] = float(np.mean(cluster_100_counts))
    return out


def positive_advantage_metrics(advantages: torch.Tensor, id2indices: dict) -> dict:
    """Per-prompt average count/fraction of generations with positive advantage."""
    pos_adv_counts: list[int] = []
    pos_adv_fracs: list[float] = []
    for batch_indices in id2indices.values():
        if not batch_indices:
            continue
        prompt_adv = advantages[torch.tensor(batch_indices, device=advantages.device, dtype=torch.long)]
        num_pos = (prompt_adv > 0).sum().item()
        pos_adv_counts.append(num_pos)
        pos_adv_fracs.append(num_pos / len(batch_indices))
    avg_num_pos = float(np.mean(pos_adv_counts)) if pos_adv_counts else 0.0
    avg_frac_pos = float(np.mean(pos_adv_fracs)) if pos_adv_fracs else 0.0
    return {
        "training_logs/grpo_avg_num_positive_adv_generations_per_prompt": avg_num_pos,
        "training_logs/grpo_avg_frac_positive_adv_generations_per_prompt": avg_frac_pos,
    }
