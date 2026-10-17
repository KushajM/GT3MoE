"""Baseline MoE routing mechanisms, on the identical backbone and expert stack.

A baseline that does not share the numerical path with the method is not a baseline, so every
router here consumes the same ``feats``/``ExpertStack`` and runs under the same ``Precision``
policy, the same optimiser, the same augmentation and the same seeds as GT3MoE.

Two honesty notes the original code raised in comments but the paper never stated:

* **Soft MoE here is not Puigcerver et al.'s slot-based Soft MoE.** The original is a soft attention
  over experts per sample, with no dispatch/combine slot matrices, because there is no token
  sequence in a CNN-MoE. That is a reasonable adaptation, but it must be named
  "Soft-Attention MoE" rather than cited as Soft MoE.
* **V-MoE's Batch Priority Routing is priority-ordered here.** The original processed the batch in
  index order and dropped whatever arrived after an expert filled, which is first-come-first-served,
  not BPR. Real BPR sorts by gate value so the *lowest*-priority tokens are the ones dropped. Both
  are available via ``vmoe_priority``; the faithful one is the default.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def switch_balance_loss(gate_probs: torch.Tensor, top_idx: torch.Tensor, K: int) -> torch.Tensor:
    oh = F.one_hot(top_idx, K).float()
    f = oh.sum(1).clamp(max=1).mean(0)
    P = gate_probs.mean(0)
    return K * (f * P).sum()


def vmoe_load_loss(gate_probs: torch.Tensor, top_idx: torch.Tensor, K: int) -> torch.Tensor:
    imp = gate_probs.sum(0).float()
    imp_loss = imp.var() / (imp.mean() ** 2 + 1e-8)
    return switch_balance_loss(gate_probs, top_idx, K) + 0.1 * imp_loss


def _gather_sel(all_logits: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
    C = all_logits.shape[-1]
    return torch.gather(all_logits, 1, top_idx.unsqueeze(-1).expand(-1, -1, C))


def route(name: str, logits_sel: torch.Tensor, all_logits: torch.Tensor, cfg,
          training: bool, labels: Optional[torch.Tensor] = None,
          phi: Optional[torch.Tensor] = None, feats: Optional[torch.Tensor] = None,
          null_logits: Optional[torch.Tensor] = None,
          tau: Optional[torch.Tensor] = None) -> Dict:
    """Dispatch. Returns out/top_idx/top_gates/gate_probs/aux.

    ``logits_sel`` are the (possibly noise-perturbed) router logits used for SELECTION.
    ``all_logits`` is [B,K,C] from the expert stack.
    ``null_logits`` is [B,N] for AdaMoE's null experts; ``tau`` is [K] for DynMoE's thresholds.

    Every branch must return ``gate_probs`` of width exactly K and ``top_idx`` valued in [0,K).
    Widening either to cover null/inactive slots breaks switch_balance_loss, _sparse_expert and
    the index_add_ accumulators in evaluate.py.
    """
    B, K, C = all_logits.shape
    dev = all_logits.device
    k = max(1, min(int(cfg.top_k), K))
    gate_probs = F.softmax(logits_sel, dim=1)

    if name in ("switch", "stablemoe", "gt3moe"):
        top_vals, top_idx = torch.topk(gate_probs, k, dim=1)
        top_gates = top_vals / (top_vals.sum(1, keepdim=True) + 1e-8)
        sel = _gather_sel(all_logits, top_idx)
        out = (top_gates.unsqueeze(-1) * sel).sum(1)
        aux = switch_balance_loss(gate_probs, top_idx, K) if name != "gt3moe" else None
        return dict(out=out, top_idx=top_idx, top_gates=top_gates, gate_probs=gate_probs, aux=aux)

    if name == "vmoe":
        top_vals, top_idx = torch.topk(gate_probs, k, dim=1)
        top_gates = top_vals / (top_vals.sum(1, keepdim=True) + 1e-8)
        cap = max(k, int((B * k / K) * cfg.capacity_factor))
        keep = torch.ones(B, k, dtype=torch.bool, device=dev)
        if cfg.vmoe_priority:
            # Real BPR: process slots in DESCENDING gate value, so low-priority tokens are dropped.
            flat_g = top_vals.reshape(-1)
            order = torch.argsort(flat_g, descending=True)
        else:
            order = torch.arange(B * k, device=dev)          # the original's index order
        flat_e = top_idx.reshape(-1)
        counts = torch.zeros(K, dtype=torch.long, device=dev)
        keep_flat = torch.ones(B * k, dtype=torch.bool, device=dev)
        for pos in order.tolist():
            e = int(flat_e[pos])
            if counts[e] >= cap:
                keep_flat[pos] = False
            else:
                counts[e] += 1
        keep = keep_flat.view(B, k)
        g = top_gates * keep.float()
        dead = ~keep.any(1)
        if dead.any():
            g[dead] = 1.0 / k                                 # fully-dropped samples fall back
        g = g / g.sum(1, keepdim=True).clamp(min=1e-8)
        sel = _gather_sel(all_logits, top_idx)
        out = (g.unsqueeze(-1) * sel).sum(1)
        return dict(out=out, top_idx=top_idx, top_gates=g, gate_probs=gate_probs,
                    aux=vmoe_load_loss(gate_probs, top_idx, K),
                    dropped_frac=float((~keep).float().mean()))

    if name == "expert_choice":
        # Experts pick their top-E samples from the batch.
        scores = F.softmax(logits_sel.t(), dim=1)              # [K,B]
        per_expert = max(1, int(B * k / K))
        top_vals, top_pos = torch.topk(scores, min(per_expert, B), dim=1)
        out = torch.zeros(B, C, device=dev, dtype=all_logits.dtype)
        gw = torch.zeros(B, K, device=dev, dtype=all_logits.dtype)
        for i in range(K):
            idx, gv = top_pos[i], top_vals[i].to(all_logits.dtype)
            out.index_add_(0, idx, gv.unsqueeze(-1) * all_logits[idx, i, :])
            gw.index_put_((idx, torch.full_like(idx, i)), gv, accumulate=True)
        tot = gw.sum(1, keepdim=True)
        out = out / tot.clamp(min=1e-8)
        orphan = tot.squeeze(1) < 1e-6
        if orphan.any():                                       # route orphans to their best expert
            be = logits_sel[orphan].argmax(1)
            out[orphan] = all_logits[orphan.nonzero(as_tuple=True)[0], be]
            gw[orphan, be] = 1.0
        _, pseudo = torch.topk(gw, min(k, K), dim=1)
        return dict(out=out, top_idx=pseudo, top_gates=torch.gather(gw, 1, pseudo),
                    gate_probs=gw / gw.sum(1, keepdim=True).clamp(min=1e-8), aux=None)

    if name == "ride":
        # RIDE-core inference: uniform average of every expert's logits, no gate, no routing
        # decision. Paired with cfg.expert_loss="per_expert" this is a genuine ensemble of
        # complete classifiers; paired with the default blend loss it would be a cooperating
        # residual stack, which is the m=8 failure mode and NOT RIDE.
        out = all_logits.mean(1)
        g = all_logits.new_full((B, K), 1.0 / K)
        _, pseudo = torch.topk(g, min(k, K), dim=1)
        return dict(out=out, top_idx=pseudo, top_gates=torch.gather(g, 1, pseudo),
                    gate_probs=g, aux=None)

    if name == "softmoe":
        # Soft-Attention MoE (see module docstring): dense combine, no sparsity, no balance loss.
        assert phi is not None
        w = F.softmax(feats @ phi.t(), dim=1)                  # [B,K]
        out = (w.unsqueeze(-1) * all_logits).sum(1)
        _, pseudo = torch.topk(w, min(k, K), dim=1)
        return dict(out=out, top_idx=pseudo, top_gates=torch.gather(w, 1, pseudo),
                    gate_probs=w, aux=None)

    if name == "hash":
        # Deterministic class->expert. Needs the label at train time; two-pass at eval.
        C_ = cfg.num_classes
        pri = torch.arange(C_, device=dev) % K
        sec = (torch.arange(C_, device=dev) + 1) % K
        if labels is not None:
            ti = torch.stack([pri[labels], sec[labels]], 1)[:, :k]
        else:
            pred = all_logits.mean(1).argmax(1)
            ti = torch.stack([pri[pred], sec[pred]], 1)[:, :k]
        g = torch.full((B, ti.shape[1]), 1.0 / ti.shape[1], device=dev, dtype=all_logits.dtype)
        out = (g.unsqueeze(-1) * _gather_sel(all_logits, ti)).sum(1)
        gp = torch.zeros(B, K, device=dev, dtype=all_logits.dtype)
        gp.scatter_(1, ti, 1.0 / ti.shape[1])
        return dict(out=out, top_idx=ti, top_gates=g, gate_probs=gp, aux=None)

    if name == "adamoe":
        # AdaMoE (Zeng et al., EMNLP Findings 2024): N *null* experts join the gate, so a
        # top-k selection that lands on null slots spends less than k real experts. Null slots
        # contribute a zero vector and their gate mass is NOT redistributed -- dropping it is
        # the adaptive-compute mechanism, not a bug.
        #
        # Adaptation, stated because the original is an LLM FFN-MoE: the gate is widened to
        # K+N only INSIDE this branch. The dict we return is still width-K, with gate_probs
        # renormalised over the real experts so the balance loss and routing entropy stay
        # well-defined. A sample whose whole top-k is null would emit a zero logit vector, so
        # it falls back to its single best real expert (the orphan rule expert_choice uses).
        assert null_logits is not None, "adamoe needs null_logits; see model.router_null"
        N = null_logits.shape[1]
        wide = torch.cat([logits_sel, null_logits.to(logits_sel.dtype)], dim=1)   # [B,K+N]
        pw = F.softmax(wide, dim=1)
        kk = max(1, min(k, K + N))
        _, widx = torch.topk(pw, kk, dim=1)
        real = widx < K                                                          # [B,kk]
        gw = torch.zeros(B, K, device=dev, dtype=all_logits.dtype)
        gw.scatter_add_(1, widx.clamp(max=K - 1),
                        (torch.gather(pw, 1, widx) * real).to(all_logits.dtype))
        none_real = ~real.any(1)
        if none_real.any():
            be = logits_sel[none_real].argmax(1)
            gw[none_real] = 0.0
            gw[none_real.nonzero(as_tuple=True)[0], be] = 1.0
        out = (gw.unsqueeze(-1) * all_logits).sum(1)
        gp = gw / gw.sum(1, keepdim=True).clamp(min=1e-8)
        _, pseudo = torch.topk(gw, min(k, K), dim=1)
        return dict(out=out, top_idx=pseudo, top_gates=torch.gather(gw, 1, pseudo),
                    gate_probs=gp, aux=switch_balance_loss(gp, pseudo, K),
                    experts_per_sample=float(real.float().sum(1).mean()))

    if name == "dynmoe":
        # DynMoE (Guo et al., ICLR 2025): "top-any" gating. Each expert carries a learnable
        # threshold and a sample activates every expert whose router logit clears it, so the
        # number of active experts is decided per sample rather than fixed at k.
        #
        # Adaptation: the original scores a token against learnable per-expert embeddings; here
        # the existing router linear already produces exactly that score, so tau is applied to
        # logits_sel directly. Expert *pruning* (DynMoE's adaptive K) is deliberately NOT
        # implemented -- it would change the parameter count between arms and break the
        # parameter-matched comparison this benchmark rests on. K is fixed and reported.
        assert tau is not None, "dynmoe needs tau; see model.router_tau"
        mask = logits_sel > tau.view(1, -1)                                       # [B,K]
        none_on = ~mask.any(1)
        if none_on.any():                                                         # top-1 fallback
            mask = mask.clone()
            mask[none_on.nonzero(as_tuple=True)[0], logits_sel[none_on].argmax(1)] = True
        gm = gate_probs * mask
        gp = gm / gm.sum(1, keepdim=True).clamp(min=1e-8)
        gpc = gp.to(all_logits.dtype)
        out = (gpc.unsqueeze(-1) * all_logits).sum(1)
        n_act = mask.float().sum(1)
        # Balance, plus a pull toward top_k active experts so the thresholds cannot drift to
        # "everything on" (which would silently make this a dense ensemble).
        aux = (switch_balance_loss(gp, torch.topk(gp, min(k, K), dim=1)[1], K)
               + cfg.dynmoe_sparsity_beta * ((n_act.mean() - k) / K) ** 2)
        _, pseudo = torch.topk(gp, min(k, K), dim=1)
        return dict(out=out, top_idx=pseudo, top_gates=torch.gather(gpc, 1, pseudo),
                    gate_probs=gp, aux=aux, experts_per_sample=float(n_act.mean()))

    raise ValueError(f"unknown router {name!r}")
