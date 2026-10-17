"""Backbone factory, batched expert stack, and the GT3MoE module.

Three deliberate departures from the original, each flag-controlled:

* ``ExpertStack`` replaces the Python loop over K ``nn.Sequential`` modules with two ``baddbmm``
  calls. The original issued 2K forward and ~4K backward kernel launches per step; on a 4-vCPU
  box at K=32 that is launch-bound, not FLOP-bound. Initialisation matches ``nn.Linear`` slice for
  slice, so ``expert_impl="loop"`` and ``"bmm"`` are the same model.
* The router returns **clean** logits alongside the noisy ones. The original added N(0, 0.1) to the
  router logits and used the resulting softmax for both top-k selection *and* the alignment loss;
  noise belongs in selection (exploration), not in the supervised target.
* ``feats`` is cast to fp32 at the head boundary, so the router, experts, gate, CE and the K x C
  competence matrix are all computed in fp32 by default.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- backbone

SCRATCH = ("", "none", "None", "random", "scratch")


def _weights(enum, weights):
    """Resolve a torchvision weights enum, or None for random init.

    ImageNet-LT's train images ARE a subset of ImageNet-1k train, and Mini/Tiny-ImageNet are built
    from it too, so a pretrained trunk has already seen the images we evaluate on. Those corpora
    have to run from scratch, which is what ``backbone_weights="none"`` selects.
    """
    return None if str(weights) in SCRATCH else getattr(enum, weights)


def make_backbone(name: str = "resnet50", weights: str = "IMAGENET1K_V2") -> Tuple[nn.Module, int]:
    import torchvision.models as tvm
    if name == "resnet50":
        base = tvm.resnet50(weights=_weights(tvm.ResNet50_Weights, weights))
        feat = nn.Sequential(*list(base.children())[:-1], nn.Flatten(1))
        return feat, 2048
    if name == "densenet201":
        # These two used to hardcode IMAGENET1K_V1 and ignore the argument entirely, so
        # backbone_weights was silently a no-op for anything but resnet50.
        base = tvm.densenet201(weights=_weights(tvm.DenseNet201_Weights, weights
                                                if weights in SCRATCH else "IMAGENET1K_V1"))
        feat = nn.Sequential(base.features, nn.ReLU(inplace=True),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        return feat, 1920
    if name == "mobilenetv2":
        base = tvm.mobilenet_v2(weights=_weights(tvm.MobileNet_V2_Weights, weights
                                                 if weights in SCRATCH else "IMAGENET1K_V1"))
        feat = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        return feat, 1280
    raise ValueError(f"unknown backbone {name!r}")


def make_backbone_split(name: str = "resnet50", weights: str = "IMAGENET1K_V2"):
    """Split the trunk so the LAST STAGE can be replicated into per-expert branches.

    Returns ``(stem, tail_prototype, stem_dim, feat_dim)``. Measured on ResNet-50 at 96x96:

        stem  (conv1..layer3)   8,543,296 params   1.204 GFLOP/img   80.2% of the forward
        layer4 (one expert)    14,964,736 params   0.297 GFLOP/img   19.8%

    That 19.8% is the entire point. With K=8 replicated layer4s and top-1 routing the forward cost
    is stem + one layer4, i.e. *identical to a dense ResNet-50*, while the model holds 128M
    parameters against the dense 23.5M -- 5.4x the capacity at the same compute, and a 58% saving
    against evaluating all eight. Compare the head-level experts this project has been running:
    0.5M params each and 0.14% of the forward, where sparse routing saves 0.87% and no amount of
    routing quality can matter.
    """
    import torchvision.models as tvm
    if name != "resnet50":
        raise ValueError(f"trunk experts are only wired for resnet50, not {name!r}")
    base = tvm.resnet50(weights=_weights(tvm.ResNet50_Weights, weights))
    ch = list(base.children())                       # conv1 bn1 relu maxpool L1 L2 L3 L4 avgpool fc
    stem = nn.Sequential(*ch[:7])                    # -> [B, 1024, h, w]
    tail = nn.Sequential(ch[7], ch[8], nn.Flatten(1))  # layer4 + avgpool + flatten -> [B, 2048]
    return stem, tail, 1024, 2048


def backbone_scope_params(backbone: nn.Module, name: str, scope: str):
    """Parameters PCGrad measures conflict over.

    ``layer4`` is 63.7% of ResNet-50's trunk parameters but only ~20% of its forward FLOPs, and
    ``autograd.grad`` prunes the backward graph to the subgraph reaching the requested inputs, so
    a layer4-scoped gradient genuinely stops early (~0.29x a full backward, measured).
    """
    if scope == "full":
        return list(backbone.parameters())
    if name != "resnet50":
        raise ValueError(f"scope {scope!r} is only defined for resnet50")
    # children(): conv1 bn1 relu maxpool layer1 layer2 layer3 layer4 avgpool flatten
    layer4 = backbone[7]
    if scope == "layer4":
        return list(layer4.parameters())
    if scope == "layer4.2":
        return list(layer4[2].parameters())
    raise ValueError(f"unknown pcgrad_scope {scope!r}")


# --------------------------------------------------------------------------- experts

class ExpertStack(nn.Module):
    """K parallel MLP heads ``Linear(F,H)-ReLU-Dropout-Linear(H,C)`` as batched tensors."""

    def __init__(self, K: int, in_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.K, self.in_dim, self.hidden, self.C = K, in_dim, hidden, num_classes
        self.W1 = nn.Parameter(torch.empty(K, in_dim, hidden))
        self.b1 = nn.Parameter(torch.empty(K, hidden))
        self.W2 = nn.Parameter(torch.empty(K, hidden, num_classes))
        self.b2 = nn.Parameter(torch.empty(K, num_classes))
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # identical to nn.Linear's default, applied per slice
        for W, b, fan_in in ((self.W1, self.b1, self.in_dim), (self.W2, self.b2, self.hidden)):
            for k in range(self.K):
                # nn.Linear stores weight as [out,in]; init on that layout then transpose in.
                w = torch.empty(W.shape[2], W.shape[1])
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                with torch.no_grad():
                    W[k].copy_(w.t())
                bound = 1.0 / math.sqrt(fan_in)
                with torch.no_grad():
                    nn.init.uniform_(b[k], -bound, bound)

    def forward_per_expert(self, fe: torch.Tensor) -> torch.Tensor:
        """fe [K,B,F] -- each expert gets its OWN features. Returns [B,K,C].

        This is the trunk-expert path: expert k reads the output of its own layer4, not a shared
        vector, so experts can differ in REPRESENTATION and not merely in decision boundary.
        """
        h = torch.baddbmm(self.b1.unsqueeze(1), fe, self.W1)
        h = self.drop(F.relu(h))
        z = torch.baddbmm(self.b2.unsqueeze(1), h, self.W2)
        return z.transpose(0, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B,F] -> [B,K,C]."""
        B = feats.shape[0]
        fe = feats.unsqueeze(0).expand(self.K, B, self.in_dim)
        h = torch.baddbmm(self.b1.unsqueeze(1), fe, self.W1)      # [K,B,H]
        h = self.drop(F.relu(h))
        z = torch.baddbmm(self.b2.unsqueeze(1), h, self.W2)       # [K,B,C]
        return z.transpose(0, 1)                                   # [B,K,C]


class ExpertLoop(nn.Module):
    """The original implementation: a ModuleList of nn.Sequential, looped in Python."""

    def __init__(self, K: int, in_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, num_classes))
            for _ in range(K)])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.stack([e(feats) for e in self.experts], dim=1)


# --------------------------------------------------------------------------- model

class GT3MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, dim = make_backbone(cfg.backbone, cfg.backbone_weights)
        cfg.feat_dim = dim
        self.K = max(1, int(cfg.num_experts))
        self.C = int(cfg.num_classes)
        self.top_k = max(1, min(int(cfg.top_k), self.K))
        self.noise = float(cfg.router_noise)
        self.router = nn.Linear(dim, self.K)
        cls = ExpertStack if cfg.expert_impl == "bmm" else ExpertLoop
        self.experts = cls(self.K, dim, cfg.expert_hidden, self.C, cfg.dropout_rate)
        self.router_type = cfg.router
        self.cfg = cfg
        # Soft-Attention MoE uses learned per-expert prototypes instead of a linear router.
        self.phi = nn.Parameter(torch.randn(self.K, dim) * 0.01) if cfg.router == "softmoe" else None
        # AdaMoE's null-expert logits and DynMoE's per-expert thresholds. Both are named
        # ``router_*`` on purpose: make_optimizer assigns learning rates by name prefix, so a
        # different name would silently train them at lr_experts (as self.phi above does).
        self.router_null = (nn.Linear(dim, max(0, int(cfg.adamoe_null_experts)))
                            if cfg.router == "adamoe" else None)
        self.router_tau = (nn.Parameter(torch.full((self.K,), float(cfg.dynmoe_tau_init)))
                           if cfg.router == "dynmoe" else None)

    def expert_parameters(self):
        return list(self.experts.parameters())

    @torch.no_grad()
    def pooled_features(self, x):
        """[B, feat_dim] trunk features, for the k-means assignment init."""
        return self.backbone(x).float()

    def forward(self, x: torch.Tensor, training: bool = True, prec=None,
                gen: Optional[torch.Generator] = None, sparse: bool = False,
                labels: Optional[torch.Tensor] = None):
        """Returns a dict; see ``keys`` below.

        ``sparse=True`` evaluates only the top-k experts (honest inference FLOPs). The training
        path needs all K, because the dense K x C competence matrix is read off the same tensor.
        """
        import contextlib
        bc = prec.backbone_autocast() if prec is not None else contextlib.nullcontext()
        hc = prec.head_autocast() if prec is not None else contextlib.nullcontext()
        with bc:
            feats = self.backbone(x)
        feats = feats.float()                                     # <-- head precision boundary
        with hc:
            logits_clean = self.router(feats)                     # [B,K] fp32
            logits_sel = logits_clean
            if training and self.noise > 0:
                n = torch.randn(logits_clean.shape, device=logits_clean.device,
                                dtype=logits_clean.dtype, generator=gen)
                logits_sel = logits_clean + n * self.noise
            gate_probs = F.softmax(logits_sel, dim=1)
            top_vals, top_idx = torch.topk(gate_probs, self.top_k, dim=1)
            top_gates = top_vals / (top_vals.sum(1, keepdim=True) + 1e-8)

            # adamoe and dynmoe are deliberately NOT in this tuple. The sparse path reuses the
            # pre-route ``top_idx`` above, which for those two is not the selection they actually
            # make (AdaMoE's top-k runs over K+N slots, DynMoE thresholds instead of ranking), so
            # it would measure FLOPs for a model that is not the one being evaluated. Their real
            # cost is dense_backbone + (experts_per_sample / K) * expert_cost, computed from the
            # metric each returns -- and expert heads are 0.14% of the forward pass each, so the
            # difference is immaterial either way.
            if self.router_type in ("gt3moe", "switch", "stablemoe") and sparse:
                # Evaluate ONLY the selected experts. Computing the dense [B,K,C] tensor here as
                # well -- which this used to do unconditionally -- made the measured "sparse"
                # forward more expensive than the dense one (1.523 vs 1.519 GFLOPs), i.e. the
                # number reported as the inference saving was really dense + gather.
                #
                # ``all_logits`` is None on this path, deliberately. Nothing reads it here: the
                # competence matrix, the best-expert oracle and the uniform-ensemble metric all
                # need every expert and so all run with sparse=False. Returning a stale dense
                # tensor instead would let a future caller silently pay for it.
                all_logits = None
                sel = self._sparse_expert(feats, top_idx)
                out = (top_gates.unsqueeze(-1) * sel).sum(1)
                aux = None
            else:
                all_logits = self.experts(feats)                  # [B,K,C]
                from .routers import route
                r = route(self.router_type, logits_sel, all_logits, self.cfg, training,
                          labels=labels, phi=self.phi, feats=feats,
                          null_logits=None if self.router_null is None else self.router_null(feats),
                          tau=self.router_tau)
                out, top_idx, top_gates = r["out"], r["top_idx"], r["top_gates"]
                gate_probs, aux = r["gate_probs"], r.get("aux")
                sel = None

        return dict(out=out, gate_probs=gate_probs, top_idx=top_idx, top_gates=top_gates,
                    all_logits=all_logits, feats=feats, router_logits=logits_clean,
                    router_logits_sel=logits_sel, sel=sel, router_aux=aux)

    def _sparse_expert(self, feats: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """Evaluate only the selected experts. Only valid for ExpertStack."""
        if not isinstance(self.experts, ExpertStack):
            raise RuntimeError("sparse inference requires expert_impl='bmm'")
        e = self.experts
        B, k = top_idx.shape
        f = feats.unsqueeze(1).expand(B, k, e.in_dim).reshape(B * k, 1, e.in_dim)
        flat = top_idx.reshape(-1)
        W1, b1 = e.W1[flat], e.b1[flat].unsqueeze(1)
        h = torch.baddbmm(b1, f, W1)
        h = F.relu(h)
        W2, b2 = e.W2[flat], e.b2[flat].unsqueeze(1)
        z = torch.baddbmm(b2, h, W2)
        return z.reshape(B, k, e.C)


class DenseHead(nn.Module):
    """Single-head control on the same backbone and the same numerical path.

    A control that does not share the precision policy, augmentation and optimiser settings with
    GT3MoE is not a control, so this deliberately reuses ``Precision`` and the same forward shape.
    Note that ``num_experts=1`` on GT3MoE is an even stricter control and costs one config flag.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone, dim = make_backbone(cfg.backbone, cfg.backbone_weights)
        cfg.feat_dim = dim
        self.head = nn.Sequential(
            nn.Linear(dim, cfg.expert_hidden), nn.ReLU(), nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.expert_hidden, cfg.num_classes))
        self.K, self.C, self.top_k = 1, int(cfg.num_classes), 1

    def expert_parameters(self):
        return list(self.head.parameters())

    @torch.no_grad()
    def pooled_features(self, x):
        return self.backbone(x).float()

    def forward(self, x, training: bool = True, prec=None, gen=None, sparse: bool = False,
                labels=None):
        import contextlib
        bc = prec.backbone_autocast() if prec is not None else contextlib.nullcontext()
        hc = prec.head_autocast() if prec is not None else contextlib.nullcontext()
        with bc:
            feats = self.backbone(x)
        feats = feats.float()
        with hc:
            out = self.head(feats)
        B = out.shape[0]
        one = torch.ones(B, 1, device=out.device)
        return dict(out=out, gate_probs=one, top_idx=torch.zeros(B, 1, dtype=torch.long,
                                                                 device=out.device),
                    top_gates=one, all_logits=out.unsqueeze(1), feats=feats,
                    router_logits=torch.zeros(B, 1, device=out.device),
                    router_logits_sel=torch.zeros(B, 1, device=out.device),
                    sel=out.unsqueeze(1), router_aux=None)


class TrunkMoE(nn.Module):
    """Experts are whole trunk BLOCKS (a replicated layer4 + head), not MLP heads on one vector.

    The point of the change. With head-level experts every expert reads the SAME 2048-d feature, so
    experts can only differ in their decision boundary -- measured consequence: each is 0.5M params
    and 0.14% of the forward, sparse routing saves 0.87%, and no routing result can matter. Here
    expert k owns its own layer4, so it computes its own REPRESENTATION, and at K=8 the model holds
    128M parameters against a dense ResNet-50's 23.5M while top-1 routing costs exactly the same
    forward FLOPs. That is the actual mixture-of-experts proposition rather than a relabelling of it.

    The router necessarily reads POOLED STEM features (1024-d from layer3), not the 2048-d vector,
    because routing has to happen before the expert computes. That is a real property of trunk-level
    MoE and a weaker signal than the head-level router enjoys; it is not a shortcut.
    """

    def __init__(self, cfg):
        super().__init__()
        import copy
        stem, tail, stem_dim, dim = make_backbone_split(cfg.backbone, cfg.backbone_weights)
        #: named ``backbone`` so make_optimizer's ``n.startswith("backbone")`` gives the shared
        #: trunk the low backbone lr rather than the expert lr.
        self.backbone = stem
        self.K = max(1, int(cfg.num_experts))
        self.C = int(cfg.num_classes)
        self.top_k = max(1, min(int(cfg.top_k), self.K))
        self.noise = float(cfg.router_noise)
        self.stem_dim, self.dim = stem_dim, dim
        cfg.feat_dim = stem_dim                       # the router / k-means space
        self.tails = nn.ModuleList([copy.deepcopy(tail) for _ in range(self.K)])
        self.router = nn.Linear(stem_dim, self.K)
        self.router_null = (nn.Linear(stem_dim, max(0, int(cfg.adamoe_null_experts)))
                            if cfg.router == "adamoe" else None)
        self.router_tau = (nn.Parameter(torch.full((self.K,), float(cfg.dynmoe_tau_init)))
                           if cfg.router == "dynmoe" else None)
        self.experts = ExpertStack(self.K, dim, cfg.expert_hidden, self.C, cfg.dropout_rate)
        self.router_type = cfg.router
        self.cfg = cfg
        self.phi = None

    def expert_parameters(self):
        return list(self.tails.parameters()) + list(self.experts.parameters())

    @torch.no_grad()
    def pooled_features(self, x):
        return F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1).float()

    def forward(self, x, training: bool = True, prec=None, gen=None, sparse: bool = False,
                labels=None):
        import contextlib
        bc = prec.backbone_autocast() if prec is not None else contextlib.nullcontext()
        hc = prec.head_autocast() if prec is not None else contextlib.nullcontext()
        with bc:
            f = self.backbone(x)                                   # [B,1024,h,w]
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1).float()     # [B,1024]
        with hc:
            logits_clean = self.router(pooled)
            logits_sel = logits_clean
            if training and self.noise > 0:
                n = torch.randn(logits_clean.shape, device=logits_clean.device,
                                dtype=logits_clean.dtype, generator=gen)
                logits_sel = logits_clean + n * self.noise
            gate_probs = F.softmax(logits_sel, dim=1)
            top_vals, top_idx = torch.topk(gate_probs, self.top_k, dim=1)
            top_gates = top_vals / (top_vals.sum(1, keepdim=True) + 1e-8)

        if sparse:
            # Honest inference cost: only the selected branches run. Used by measure_flops.
            sel = []
            for j in range(self.top_k):
                k = int(top_idx[0, j])
                with bc:
                    tf = self.tails[k](f).float()
                sel.append(self.experts.forward_per_expert(
                    tf.unsqueeze(0).expand(self.K, *tf.shape))[:, k, :])
            out = (top_gates.unsqueeze(-1) * torch.stack(sel, 1)).sum(1)
            return dict(out=out, gate_probs=gate_probs, top_idx=top_idx, top_gates=top_gates,
                        all_logits=None, feats=pooled, router_logits=logits_clean,
                        router_logits_sel=logits_sel, sel=None, router_aux=None)

        with bc:
            fe = torch.stack([t(f) for t in self.tails], 0)         # [K,B,2048]
        with hc:
            all_logits = self.experts.forward_per_expert(fe.float())   # [B,K,C]
            from .routers import route
            r = route(self.router_type, logits_sel, all_logits, self.cfg, training,
                      labels=labels, phi=None, feats=pooled,
                      null_logits=None if self.router_null is None else self.router_null(pooled),
                      tau=self.router_tau)
        return dict(out=r["out"], gate_probs=r["gate_probs"], top_idx=r["top_idx"],
                    top_gates=r["top_gates"], all_logits=all_logits, feats=pooled,
                    router_logits=logits_clean, router_logits_sel=logits_sel, sel=None,
                    router_aux=r.get("aux"))


def build_model(cfg) -> nn.Module:
    if getattr(cfg, "trunk_experts", False) and not cfg.dense_head:
        m = TrunkMoE(cfg)
    else:
        m = DenseHead(cfg) if cfg.dense_head else GT3MoE(cfg)
    if cfg.channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def count_params(m: nn.Module) -> dict:
    tot = sum(p.numel() for p in m.parameters())
    bb = sum(p.numel() for p in m.backbone.parameters())
    return dict(total=tot, backbone=bb, heads=tot - bb)
