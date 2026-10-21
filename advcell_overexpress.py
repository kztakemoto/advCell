#!/usr/bin/env python3
"""
Overexpress helper (pop-and-insert) following the official Geneformer implementation
============================================================
The official InSilicoPerturber overexpress removes the target gene from its original position in
the rank value encoding and then inserts it at the front (pop-and-insert).
Merely prepending makes the same token appear twice in the sequence, an input outside the pretraining distribution.

This module pads the variable-length sequences and returns them together with
attention_mask.

Main functions:
    overexpress_batch(orig_ids, cand_tokens, max_seq_len) -> (ids, attn, n_removed)
    build_inputs_embeds(E, ids, k_adv, adv_emb)           -> (B, T, D) tensor
    masked_mean_pool(hidden, attn)                        -> (B, D) tensor

Self-test:
    python advcell_overexpress.py
"""

from typing import Tuple

import torch


PAD_TOKEN_ID = 0

# Overexpress mode
#   "pop_and_insert" : follows the official InSilicoPerturber. If already present, remove from the original position and insert at the front.
#   "prepend_only"   : Prepends without removal, so duplicate tokens arise.
#                      Use only for A/B comparison against pop_and_insert.
_MODE = "pop_and_insert"
VALID_MODES = ("pop_and_insert", "prepend_only")


def set_overexpress_mode(mode: str) -> None:
    """Set the default overexpress mode (meant to be called once from main)."""
    global _MODE
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    _MODE = mode


def get_overexpress_mode() -> str:
    return _MODE


def overexpress_batch(
    orig_ids: torch.Tensor,
    cand_tokens: torch.Tensor,
    max_seq_len: int,
    pad_id: int = PAD_TOKEN_ID,
    mode: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply overexpress to a batch of candidates.

    Args:
        orig_ids:    (L,)   long. The cell's rank value encoding (padding removed).
        cand_tokens: (B, k) long. Tokens to overexpress in each candidate.
                     Assumes no duplicates within a row.
        max_seq_len: upper limit of the model's input length (2048 for V1).
        pad_id:      padding token ID.
        mode:        if None, follow the setting of set_overexpress_mode.

    Returns:
        out_ids:    (B, T) long. The first k columns are cand_tokens, the rest is the remaining sequence.
        attn:       (B, T) long. 1 at valid positions.
        n_removed:  (B,)   long. Number of tokens removed from the original sequence
                    (= number of inserted TFs that were originally expressed; for diagnostics).
                    For prepend_only it means "the number of duplicates"; nothing is removed.
    """
    if orig_ids.dim() != 1:
        raise ValueError(f"orig_ids must be 1-D, got {tuple(orig_ids.shape)}")
    if cand_tokens.dim() != 2:
        raise ValueError(f"cand_tokens must be 2-D, got {tuple(cand_tokens.shape)}")

    mode = _MODE if mode is None else mode
    device = orig_ids.device
    L = orig_ids.numel()
    B, k = cand_tokens.shape

    # for each candidate, which positions of the original sequence match the target tokens
    hit = (orig_ids.view(1, 1, L) == cand_tokens.view(B, k, 1)).any(dim=1)
    n_removed = hit.sum(dim=1)

    if mode == "prepend_only":
        # no removal; cut k from the tail and prepend.
        base = orig_ids[: max(max_seq_len - k, 0)]
        out_ids = torch.cat(
            [cand_tokens, base.view(1, -1).expand(B, -1)], dim=1
        ).contiguous()
        attn = torch.ones_like(out_ids)
        return out_ids, attn, n_removed

    keep = ~hit
    n_keep = keep.sum(dim=1)                           # (B,)

    # where the kept elements go (positions after leaving the first k columns free)
    pos = keep.cumsum(dim=1) - 1                       # (B, L)
    W = k + L + 1                                      # the last column is used as a dump
    dest = torch.where(keep, pos + k, torch.full_like(pos, W - 1))

    out_ids = torch.full((B, W), pad_id, dtype=torch.long, device=device)
    out_ids.scatter_(1, dest, orig_ids.view(1, L).expand(B, L))
    out_ids[:, :k] = cand_tokens

    # fix the length and truncate
    lengths = (n_keep + k).clamp(max=max_seq_len)      # (B,)
    T = int(lengths.max().item())
    out_ids = out_ids[:, :T].contiguous()

    ar = torch.arange(T, device=device).view(1, T)
    attn = (ar < lengths.view(B, 1)).long()
    out_ids = out_ids * attn + pad_id * (1 - attn)     # fill the truncated part with pad

    return out_ids, attn, n_removed


def overexpress_single(
    orig_ids: torch.Tensor,
    tokens: torch.Tensor,
    max_seq_len: int,
    pad_id: int = PAD_TOKEN_ID,
    mode: str = None,
):
    """Single-sequence version. If tokens is empty, return the original sequence as it is."""
    if tokens.numel() == 0:
        ids = orig_ids[:max_seq_len].view(1, -1)
        attn = torch.ones_like(ids)
        return ids, attn, torch.zeros(1, dtype=torch.long, device=orig_ids.device)
    return overexpress_batch(
        orig_ids, tokens.view(1, -1), max_seq_len, pad_id=pad_id, mode=mode
    )


def build_inputs_embeds(
    E: torch.Tensor,
    ids: torch.Tensor,
    k_adv: int,
    adv_emb: torch.Tensor = None,
) -> torch.Tensor:
    """Assemble inputs_embeds.

    Args:
        E:       (V, D) word embedding matrix.
        ids:     (B, T) long. Output of overexpress_batch.
        k_adv:   number of leading adversarial positions.
        adv_emb: (k, D) or (B, k, D). Leaf tensor through which the gradient should flow.
                 If None, use E[ids[:, :k_adv]] (no gradient).

    Returns:
        (B, T, D)
    """
    B = ids.shape[0]
    with torch.no_grad():
        rest = E[ids[:, k_adv:]]                       # (B, T-k, D)
    rest = rest.detach()

    if adv_emb is None:
        with torch.no_grad():
            head = E[ids[:, :k_adv]].detach()
    else:
        head = adv_emb
        if head.dim() == 2:
            head = head.unsqueeze(0).expand(B, -1, -1)
    return torch.cat([head, rest], dim=1)


def masked_mean_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Mean pool excluding the padding.

    Args:
        hidden: (B, T, D)
        attn:   (B, T)
    Returns:
        (B, D)
    """
    m = attn.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK ' if cond else 'NG '}] {name}")
        ok = ok and cond

    print("test 1: duplicates are removed and the order is kept")
    orig = torch.tensor([10, 11, 12, 13, 14])
    cand = torch.tensor([[12, 99]])          # 12 is already present, 99 is new
    ids, attn, nrem = overexpress_batch(orig, cand, max_seq_len=100)
    check("inserted at the front", ids[0, :2].tolist() == [12, 99])
    check("the original 12 is gone", ids[0, 2:].tolist()[:4] == [10, 11, 13, 14])
    check("length = 2 + 4", int(attn[0].sum()) == 6)
    check("n_removed = 1", int(nrem[0]) == 1)
    check("12 appears only once in the sequence", (ids[0][:int(attn[0].sum())] == 12).sum().item() == 1)

    print("test 2: padding when the length differs between candidates")
    cand = torch.tensor([[12, 99], [98, 99], [10, 14]])
    ids, attn, nrem = overexpress_batch(orig, cand, max_seq_len=100)
    check("n_removed = [1, 0, 2]", nrem.tolist() == [1, 0, 2])
    check("lengths = [6, 7, 5]", attn.sum(1).tolist() == [6, 7, 5])
    check("T = 7", ids.shape[1] == 7)
    check("the tail of row 0 is pad", int(ids[0, 6]) == PAD_TOKEN_ID)
    check("the last 2 of row 2 are pad", ids[2, 5:].tolist() == [PAD_TOKEN_ID] * 2)
    for b in range(3):
        n = int(attn[b].sum())
        seq = ids[b][:n].tolist()
        check(f"no duplicates in row {b}", len(set(seq)) == len(seq))

    print("test 3: truncation at max_seq_len")
    orig = torch.arange(10, 20)              # length 10
    cand = torch.tensor([[99, 98]])          # both new -> should become 12
    ids, attn, nrem = overexpress_batch(orig, cand, max_seq_len=8)
    check("length is cut to 8", int(attn[0].sum()) == 8)
    check("the front is the candidate", ids[0, :2].tolist() == [99, 98])
    check("the tail is dropped", ids[0, 2:8].tolist() == [10, 11, 12, 13, 14, 15])

    print("test 4: when removal frees a slot, the trailing part survives")
    orig = torch.arange(10, 20)
    cand = torch.tensor([[10, 11]])          # both already present -> length stays 10
    ids, attn, nrem = overexpress_batch(orig, cand, max_seq_len=10)
    check("length 10", int(attn[0].sum()) == 10)
    check("n_removed = 2", int(nrem[0]) == 2)
    check("content", ids[0].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    print("test 5: gradient of build_inputs_embeds")
    V, D, k = 50, 8, 2
    E = torch.randn(V, D, requires_grad=True)
    orig = torch.arange(10, 30)
    cand = torch.tensor([[12, 45], [46, 47]])
    ids, attn, _ = overexpress_batch(orig, cand, max_seq_len=64)
    adv = E[cand[0]].detach().clone().requires_grad_(True)
    emb = build_inputs_embeds(E, ids[:1], k, adv_emb=adv)
    pooled = masked_mean_pool(emb, attn[:1])
    pooled.sum().backward()
    check("the gradient reaches adv_emb", adv.grad is not None)
    check("the gradient is nonzero", adv.grad is not None and adv.grad.abs().sum() > 0)
    check("no gradient leaks into E", E.grad is None or E.grad.abs().sum() == 0)

    print("test 6: masked_mean_pool ignores the padding")
    h = torch.ones(1, 5, 3)
    h[0, 3:] = 999.0
    a = torch.tensor([[1, 1, 1, 0, 0]])
    check("mean excluding pad = 1", torch.allclose(masked_mean_pool(h, a),
                                                  torch.ones(1, 3)))

    print("test 7: prepend_only prepends without removal")
    orig = torch.arange(10, 30)              # length 20
    cand = torch.tensor([[12, 99]])          # 12 is already present
    ids, attn, nrem = overexpress_batch(orig, cand, max_seq_len=18,
                                        mode="prepend_only")
    expected = [12, 99] + list(range(10, 26))   # prepend + cut k from the tail
    check("same sequence as the old implementation", ids[0].tolist() == expected)
    check("all positions valid", int(attn[0].sum()) == 18)
    check("12 appears twice (the duplicate is reproduced)",
          (ids[0] == 12).sum().item() == 2)
    check("n_removed reports the number of duplicates", int(nrem[0]) == 1)
    check("the default mode is pop_and_insert", get_overexpress_mode() == "pop_and_insert")

    print("\n" + ("All tests passed" if ok else "*** some tests failed ***"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
