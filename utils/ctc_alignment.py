# CTC Forced Alignment
# From nvasr/utils/ctc_alignment.py

import torch

def ctc_forced_align(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    ignore_id: int = -1,
) -> torch.Tensor:
    """Align a CTC label sequence to an emission.

    Args:
        log_probs: Log probability of CTC emission output. Shape (B, T, C).
        targets: Target sequence. Shape (B, L).
        input_lengths: Lengths of the inputs. Shape (B,).
        target_lengths: Lengths of the targets. Shape (B,).
        blank: Index of blank symbol. Default: 0.
        ignore_id: Index of ignore symbol. Default: -1.
    """
    targets[targets == ignore_id] = blank

    batch_size, input_time_size, _ = log_probs.size()
    bsz_indices = torch.arange(batch_size, device=input_lengths.device)

    _t_a_r_g_e_t_s_ = torch.cat(
        (
            torch.stack((torch.full_like(targets, blank), targets), dim=-1).flatten(start_dim=1),
            torch.full_like(targets[:, :1], blank),
        ),
        dim=-1,
    )
    diff_labels = torch.cat(
        (
            torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
            _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2],
        ),
        dim=1,
    )

    neg_inf = torch.tensor(float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
    padding_num = 2
    padded_t = padding_num + _t_a_r_g_e_t_s_.size(-1)
    best_score = torch.full((batch_size, padded_t), neg_inf, device=log_probs.device, dtype=log_probs.dtype)
    best_score[:, padding_num + 0] = log_probs[:, 0, blank]
    best_score[:, padding_num + 1] = log_probs[bsz_indices, 0, _t_a_r_g_e_t_s_[:, 1]]

    backpointers = torch.zeros((batch_size, input_time_size, padded_t), device=log_probs.device, dtype=targets.dtype)

    for t in range(1, input_time_size):
        prev = torch.stack(
            (best_score[:, 2:], best_score[:, 1:-1], torch.where(diff_labels, best_score[:, :-2], neg_inf))
        )
        prev_max_value, prev_max_idx = prev.max(dim=0)
        best_score[:, padding_num:] = log_probs[:, t].gather(-1, _t_a_r_g_e_t_s_) + prev_max_value
        backpointers[:, t, padding_num:] = prev_max_idx

    l1l2 = best_score.gather(
        -1, torch.stack((padding_num + target_lengths * 2 - 1, padding_num + target_lengths * 2), dim=-1)
    )

    path = torch.zeros((batch_size, input_time_size), device=best_score.device, dtype=torch.long)
    path[bsz_indices, input_lengths - 1] = padding_num + target_lengths * 2 - 1 + l1l2.argmax(dim=-1)

    for t in range(input_time_size - 1, 0, -1):
        target_indices = path[:, t]
        prev_max_idx = backpointers[bsz_indices, t, target_indices]
        path[:, t - 1] += target_indices - prev_max_idx

    alignments = _t_a_r_g_e_t_s_.gather(dim=-1, index=(path - padding_num).clamp(min=0))
    return alignments
