"""CIFAR-100-LT construction and a GPU-resident data pipeline.

Two things here matter beyond plumbing.

1. **Augmentation.** The original cached ``torch.stack([tfm(i) for i in imgs])`` in
   ``CachedDataset.__init__`` and ``__getitem__`` returned ``self.x[i]`` untransformed, so
   ``RandomHorizontalFlip`` and ``RandomCrop(96, padding=4)`` were applied exactly ONCE. Every
   published CIFAR-100-LT run trained 60 epochs on a single fixed augmented copy of 9,207 images.
   ``augment="frozen_once"`` reproduces that; ``"per_batch"`` is the fix.

2. **Throughput.** ml.g5.xlarge has 4 vCPUs. A per-item PIL pipeline
   (ToPILImage -> Resize(96) -> flip -> crop -> ToTensor -> Normalize) starves an A10G. We resize
   once on CPU, keep uint8 on the GPU (train 9,207 x 3 x 96 x 96 = 254 MB, test 276 MB) and do
   crop/flip/normalise as batched GPU ops. There is no DataLoader in the training path at all.

The long-tail protocol follows Cui et al. exactly as the original implemented it: a stratified
``val_frac`` split is carved from the *balanced* train pool first, then exponential decay is applied
to the remaining 85%. At rho=100 that yields n_max=425, n_min=4, N=9207 and a 31/35/34
Many/Medium/Few split, which reproduces the paper's Section 3.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# OLTR thresholds, as the original used them.
MANY_MIN = 100      # n > 100
FEW_MAX = 20        # n < 20


# --------------------------------------------------------------------------- construction

def _resize_uint8(imgs: np.ndarray, size: int) -> np.ndarray:
    """Resize [N,H,W,3] uint8 -> [N,size,size,3] uint8 with PIL bilinear.

    Matches ``transforms.Resize((size, size))`` on a PIL image, which is what the original did.
    """
    from PIL import Image
    if imgs.ndim == 4 and imgs.shape[1] == size and imgs.shape[2] == size:
        return imgs          # loader already emitted this size; a PIL round-trip would be a
                             # slow no-op (185k images for ImageNet-LT)
    out = np.empty((len(imgs), size, size, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        out[i] = np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR))
    return out


def class_splits(counts: np.ndarray) -> Tuple[list, list, list]:
    many = [int(c) for c in range(len(counts)) if counts[c] > MANY_MIN]
    med = [int(c) for c in range(len(counts)) if FEW_MAX <= counts[c] <= MANY_MIN]
    few = [int(c) for c in range(len(counts)) if counts[c] < FEW_MAX]
    return many, med, few


def _load_tiny_imagenet(root: str):
    """Tiny-ImageNet-200 from the official zip -> (x_train, y_train, x_test, y_test) uint8 NHWC.

    The official *test* split is unlabelled, so the labelled **val** split (50 images/class, given
    by ``val_annotations.txt``) is used as the balanced test set. Classes are taken in sorted wnid
    order so the label map is deterministic across machines.
    """
    import zipfile
    from PIL import Image
    zp = os.path.join(root, "tiny-imagenet-200.zip")
    if not os.path.exists(zp):
        raise FileNotFoundError(f"{zp} missing: fetch tiny-imagenet-200.zip into {root}")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        wnids = sorted({n.split("/")[2] for n in names
                        if n.startswith("tiny-imagenet-200/train/") and n.count("/") > 2})
        lab = {w: i for i, w in enumerate(wnids)}
        xs, ys = [], []
        for n in names:
            if "/train/" in n and n.endswith(".JPEG"):
                xs.append(np.asarray(Image.open(z.open(n)).convert("RGB")))
                ys.append(lab[n.split("/")[2]])
        ann = z.read("tiny-imagenet-200/val/val_annotations.txt").decode().splitlines()
        # whitespace split: the file is <filename> TAB <wnid> TAB <bbox...>
        vmap = {f[0]: f[1] for f in (l.split() for l in ann) if len(f) >= 2}
        xt, yt = [], []
        for n in names:
            if "/val/images/" in n and n.endswith(".JPEG"):
                xt.append(np.asarray(Image.open(z.open(n)).convert("RGB")))
                yt.append(lab[vmap[os.path.basename(n)]])
    return (np.stack(xs), np.array(ys, dtype=np.int64),
            np.stack(xt), np.array(yt, dtype=np.int64))


def _load_mini_imagenet(root: str, side: int = 84):
    """MiniImageNet from the ``timm/mini-imagenet`` parquet shards.

    That mirror splits by IMAGE over all 100 classes (train 100x500, validation 100x100); other
    mirrors ship the few-shot 64/16/20 CLASS split, which would silently hand back a 64-class
    problem. Images are full-resolution JPEG bytes, decoded and resized to ``side`` here so the
    cached npz stays small; ``_pack`` then takes them to ``img_size``.
    """
    import glob
    import io as _io
    import pyarrow.parquet as pq
    from PIL import Image
    d = os.path.join(root, "mini")

    def read(split):
        xs, ys = [], []
        for f in sorted(glob.glob(os.path.join(d, f"{split}-*.parquet"))):
            # Streamed: a shard is ~480 MB of full-resolution JPEG and read_table().to_pylist()
            # would hold the encoded bytes, the Python list and the decoded array at once.
            for rb in pq.ParquetFile(f).iter_batches(batch_size=256,
                                                     columns=["image", "label"]):
                for im, y in zip(rb.column("image").to_pylist(),
                                 rb.column("label").to_pylist()):
                    b = im["bytes"] if isinstance(im, dict) else im
                    pic = Image.open(_io.BytesIO(b)).convert("RGB").resize((side, side),
                                                                          Image.BILINEAR)
                    xs.append(np.asarray(pic))
                    ys.append(int(y))
        if not xs:
            raise FileNotFoundError(f"no {split}-*.parquet under {d}")
        return np.stack(xs), np.array(ys, dtype=np.int64)

    xa, ya = read("train")
    xt, yt = read("validation")          # 100 x 100, balanced -> used as the test set
    return xa, ya, xt, yt


#: Corpora whose train/val/test membership is fixed by published split files rather than derived
#: here. These bypass BOTH the stratified val carve and the exponential long-tail: their imbalance
#: is already in the split, and re-deriving it would silently change n_min and the Many/Med/Few
#: partition (a 15% val carve on ImageNet-LT drops n_min from 5 to 4).
PRESPLIT = ("imagenet_lt",)


def _load_imagenet_lt(root: str, side: int = 96):
    """ImageNet-LT (Liu et al.) -> the six arrays, already split, at ``side`` resolution.

    Train and val are drawn from ImageNet-1k *train*; test is the full balanced ImageNet-1k
    validation set (50 per class). Membership comes from the official
    ``ImageNet_LT_{train,val,test}.txt``; pixels come from a 256px parquet mirror whose
    ``image.path`` preserves the original filenames, so the mapping is exact rather than a
    same-distribution resample.

    Set ``GT3MOE_IMAGENET_DIR`` to a directory holding ``splits/*.txt`` and ``data/*.parquet``.
    """
    import glob
    import io as _io
    import pyarrow.parquet as pq
    from PIL import Image

    d = os.environ.get("GT3MOE_IMAGENET_DIR", os.path.join(root, "imagenet_lt"))
    want, counts = {}, [0, 0, 0]
    for sid, split in enumerate(("train", "val", "test")):
        f = os.path.join(d, "splits", f"ImageNet_LT_{split}.txt")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} missing; see GT3MOE_IMAGENET_DIR")
        for line in open(f):
            path, lab = line.rsplit(" ", 1)
            stem = os.path.basename(path).rsplit(".", 1)[0]
            want[stem] = (sid, counts[sid], int(lab))
            counts[sid] += 1

    xs = [np.empty((n, side, side, 3), dtype=np.uint8) for n in counts]
    ys = [np.empty(n, dtype=np.int64) for n in counts]
    seen, mismatched = 0, 0
    # Recursive: a snapshot_download lands the shards under <repo>/data/, not <dir>/data/.
    shards = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"no *.parquet under {d}; see GT3MOE_IMAGENET_DIR")
    for f in shards:
        for rb in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "label"]):
            for im, plab in zip(rb.column("image").to_pylist(), rb.column("label").to_pylist()):
                # mirror names are "<original stem>_<wnid>.JPEG"; the wnid always starts with 'n'
                # and original stems never end in one, so a single rsplit recovers the stem.
                stem = os.path.basename(im["path"]).rsplit(".", 1)[0].rsplit("_n", 1)[0]
                hit = want.get(stem)
                if hit is None:
                    continue
                sid, row, lab = hit
                mismatched += int(lab != int(plab))
                pic = Image.open(_io.BytesIO(im["bytes"])).convert("RGB")
                if pic.size != (side, side):
                    pic = pic.resize((side, side), Image.BILINEAR)
                xs[sid][row] = np.asarray(pic)
                ys[sid][row] = lab
                seen += 1
    if seen != sum(counts):
        raise RuntimeError(f"matched {seen} of {sum(counts)} split images; parquet mirror "
                           f"does not cover the official ImageNet-LT split")
    if mismatched:
        raise RuntimeError(f"{mismatched} images disagree on label between the split file and "
                           f"the parquet mirror; class orderings differ")
    return xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]


def _load_cifar100_raw(root: str):
    """Raw balanced CIFAR-100, so ``build_generic_lt`` can serve the rho=1 control."""
    import torchvision
    tr = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    return (np.asarray(tr.data), np.asarray(tr.targets, dtype=np.int64),
            np.asarray(te.data), np.asarray(te.targets, dtype=np.int64))


def _long_tail(xr, yr, num_classes, rho, data_seed):
    """Exponential long-tail subsample, identical protocol to build_cifar100_lt (Cui et al.).

    rho == 1 leaves the pool untouched, which is how the balanced control is expressed.
    """
    if rho <= 1.0:
        return xr, yr
    mu = (1.0 / rho) ** (1.0 / (num_classes - 1))
    ratios = np.array([mu ** i for i in range(num_classes)])
    rng = np.random.default_rng(data_seed)
    sel = []
    for c in range(num_classes):
        idx = np.where(yr == c)[0]
        if len(idx) == 0:
            continue
        k = max(1, int(ratios[c] * len(idx)))
        sel.append(rng.permutation(idx)[:k])
    sel = np.concatenate(sel)
    rng.shuffle(sel)
    return xr[sel], yr[sel]


def _pack(xr, yr, xv, yv, xt, yt, num_classes, img_size, smoke):
    if smoke:
        keep = np.concatenate([np.where(yr == c)[0][:6] for c in range(num_classes)
                               if (yr == c).any()])
        xr, yr = xr[keep], yr[keep]
        kv = np.concatenate([np.where(yv == c)[0][:2] for c in range(num_classes)
                             if (yv == c).any()])
        xv, yv = xv[kv], yv[kv]
        kt = np.concatenate([np.where(yt == c)[0][:2] for c in range(num_classes)
                             if (yt == c).any()])
        xt, yt = xt[kt], yt[kt]
    cc = np.bincount(yr, minlength=num_classes).astype(np.int64)
    many, med, few = class_splits(cc)
    return dict(xtr=_resize_uint8(xr, img_size), ytr=yr.astype(np.int64),
                xva=_resize_uint8(xv, img_size), yva=yv.astype(np.int64),
                xte=_resize_uint8(xt, img_size), yte=yt.astype(np.int64),
                class_counts=cc, many=np.array(many), medium=np.array(med), few=np.array(few))


def build_generic_lt(root: str, dataset: str, num_classes: int, rho: float, val_frac: float,
                     img_size: int, data_seed: int, cache: bool = True,
                     smoke: bool = False, cap_per_class: int = 0) -> Dict:
    """Long-tailed builder for the non-CIFAR corpora, following the same protocol throughout.

    In every case the validation split is carved from the BALANCED pool *before* long-tailing and
    the test set is left balanced, exactly as in ``build_cifar100_lt``, so the three datasets are
    directly comparable and ``probe_split="val"`` remains an ablation rather than the default.

    * ``tinyimagenet_lt``  -- 200 classes x 500 train images, 64x64. C/K = 25 at K=8, double
      CIFAR-100's 12.5, which is the axis the m-sweep showed matters.
    * ``miniimagenet_lt``  -- 100 classes; the ``timm/mini-imagenet`` mirror is an IMAGE-level
      split over all 100 classes (train 100x500, validation 100x100), NOT the few-shot 64/16/20
      CLASS split that other mirrors ship, which would silently give a 64-class problem.
    * ``cifar100``         -- rho passed through, so rho=1 is the balanced control on the same
      pipeline, same augmentation, same everything.
    """
    from sklearn.model_selection import train_test_split

    tag = (f"{dataset}_r{rho:g}_v{val_frac:g}_s{img_size}_d{data_seed}"
           f"{f'_c{cap_per_class}' if cap_per_class else ''}{'_smoke' if smoke else ''}")
    cpath = os.path.join(root, f"{tag}.npz")
    if cache and os.path.exists(cpath):
        z = np.load(cpath)
        return {k: z[k] for k in z.files}

    if dataset in PRESPLIT:
        # Already split and already long-tailed: go straight to _pack.
        xr, yr, xv, yv, xt, yt = _load_imagenet_lt(root, side=img_size)
        d = _pack(xr, yr, xv, yv, xt, yt, num_classes, img_size, smoke)
        if cache:
            os.makedirs(root, exist_ok=True)
            np.savez_compressed(cpath, **d)
        return d

    if dataset == "tinyimagenet_lt":
        xa, ya, xt, yt = _load_tiny_imagenet(root)
    elif dataset == "miniimagenet_lt":
        xa, ya, xt, yt = _load_mini_imagenet(root)
    elif dataset == "cifar100":
        xa, ya, xt, yt = _load_cifar100_raw(root)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    xr, xv, yr, yv = train_test_split(xa, ya, test_size=val_frac,
                                      random_state=data_seed, stratify=ya)
    xr, yr = _long_tail(xr, yr, num_classes, rho, data_seed)
    if cap_per_class:
        rng = np.random.default_rng(data_seed + 1)
        keep = np.concatenate([rng.permutation(np.where(yr == c)[0])[:cap_per_class]
                               for c in range(num_classes) if (yr == c).any()])
        rng.shuffle(keep)
        xr, yr = xr[keep], yr[keep]
    d = _pack(xr, yr, xv, yv, xt, yt, num_classes, img_size, smoke)
    if cache:
        os.makedirs(root, exist_ok=True)
        np.savez_compressed(cpath, **d)
    return d


def build_cifar100_lt(root: str, num_classes: int = 100, rho: float = 100.0,
                      val_frac: float = 0.15, img_size: int = 96,
                      data_seed: int = 1234, cache: bool = True,
                      smoke: bool = False) -> Dict:
    """Return resized uint8 arrays and labels for train/val/test.

    ``val`` is carved from the balanced pool *before* long-tailing, so it holds ~75 images per
    class for all 100 classes. That makes it a balanced held-out set; using it to drive the
    assignment leaks balanced label information, which is why ``probe_split="val"`` is an
    ablation and not the default.
    """
    import torchvision
    from sklearn.model_selection import train_test_split

    tag = f"c100lt_r{rho:g}_v{val_frac:g}_s{img_size}_d{data_seed}{'_smoke' if smoke else ''}"
    cpath = os.path.join(root, f"{tag}.npz")
    if cache and os.path.exists(cpath):
        z = np.load(cpath)
        d = {k: z[k] for k in z.files}
    else:
        tr = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        te = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
        xa, ya = np.asarray(tr.data), np.asarray(tr.targets)
        xt, yt = np.asarray(te.data), np.asarray(te.targets)

        # 1. stratified val split from the BALANCED pool (random_state == data_seed)
        xr, xv, yr, yv = train_test_split(
            xa, ya, test_size=val_frac, random_state=data_seed, stratify=ya)

        # 2. exponential long-tail on the remaining 85%
        mu = (1.0 / rho) ** (1.0 / (num_classes - 1))
        ratios = np.array([mu ** i for i in range(num_classes)])
        rng = np.random.default_rng(data_seed)
        sel = []
        for c in range(num_classes):
            idx = np.where(yr == c)[0]
            k = max(1, int(ratios[c] * len(idx)))
            sel.append(rng.permutation(idx)[:k])
        sel = np.concatenate(sel)
        rng.shuffle(sel)
        xr, yr = xr[sel], yr[sel]

        if smoke:   # keep <=6 images/class and a tiny test set, for CPU CI
            keep = np.concatenate([np.where(yr == c)[0][:6] for c in range(num_classes)])
            xr, yr = xr[keep], yr[keep]
            kv = np.concatenate([np.where(yv == c)[0][:2] for c in range(num_classes)])
            xv, yv = xv[kv], yv[kv]
            kt = np.concatenate([np.where(yt == c)[0][:2] for c in range(num_classes)])
            xt, yt = xt[kt], yt[kt]

        d = dict(
            xtr=_resize_uint8(xr, img_size), ytr=yr.astype(np.int64),
            xva=_resize_uint8(xv, img_size), yva=yv.astype(np.int64),
            xte=_resize_uint8(xt, img_size), yte=yt.astype(np.int64),
        )
        if cache:
            os.makedirs(root, exist_ok=True)
            np.savez_compressed(cpath, **d)

    counts = np.bincount(d["ytr"], minlength=num_classes).astype(np.int64)
    many, med, few = class_splits(counts)
    d["class_counts"] = counts
    d["many"], d["medium"], d["few"] = np.array(many), np.array(med), np.array(few)
    return d


# --------------------------------------------------------------------------- GPU pipeline

class GPUImages:
    """uint8 images resident on the device, with batched GPU crop/flip/normalise.

    Geometry reproduces the original: Resize(S) has already happened, then
    RandomHorizontalFlip, then RandomCrop(S, padding=4) with zero fill.
    """

    def __init__(self, x_uint8_nhwc: np.ndarray, y: np.ndarray, device: str = "cuda",
                 pad: int = 4, channels_last: bool = False,
                 rrc_min_scale: float = 0.35, jitter_strength: float = 0.4):
        self.rrc_min_scale = rrc_min_scale
        self.jitter_strength = jitter_strength
        x = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).permute(0, 3, 1, 2).contiguous()
        self.x = x.to(device)                       # [N,3,S,S] uint8
        self.y = torch.from_numpy(y).to(device).long()
        self.n = self.x.shape[0]
        self.size = self.x.shape[-1]
        self.pad = pad
        self.device = device
        self.channels_last = channels_last
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self._frozen: Optional[Dict[str, torch.Tensor]] = None

    # -------------------------------------------------------------- augmentation
    def freeze_augmentation(self, gen: torch.Generator):
        """Draw one crop offset and flip bit per image and reuse them forever (the v1 bug)."""
        span = 2 * self.pad + 1
        self._frozen = dict(
            ox=torch.randint(0, span, (self.n,), device=self.device, generator=gen),
            oy=torch.randint(0, span, (self.n,), device=self.device, generator=gen),
            fl=torch.rand(self.n, device=self.device, generator=gen) < 0.5,
        )

    def _augment(self, idx: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
        xb = self.x[idx]                                        # [b,3,S,S] uint8
        if mode == "none":
            return xb
        b = xb.shape[0]
        if mode == "frozen_once":
            assert self._frozen is not None, "call freeze_augmentation() first"
            ox, oy, fl = (self._frozen["ox"][idx], self._frozen["oy"][idx],
                          self._frozen["fl"][idx])
        else:
            span = 2 * self.pad + 1
            ox = torch.randint(0, span, (b,), device=self.device, generator=gen)
            oy = torch.randint(0, span, (b,), device=self.device, generator=gen)
            fl = torch.rand(b, device=self.device, generator=gen) < 0.5
        # flip first, then pad+crop -- the original transform order
        xb = torch.where(fl.view(-1, 1, 1, 1), xb.flip(-1), xb)
        S, P = self.size, self.pad
        xp = torch.zeros((b, 3, S + 2 * P, S + 2 * P), dtype=xb.dtype, device=self.device)
        xp[:, :, P:P + S, P:P + S] = xb
        # Per-sample window via advanced indexing on a channels-last view. Index tensors are
        # [b,S,1] and [b,1,S] and broadcast, so only the output is materialised -- a gather with
        # a fully expanded [b,3,S,W] int64 index would allocate ~300 MB at b=128.
        ar = torch.arange(S, device=self.device)
        rows = (oy.view(b, 1, 1) + ar.view(1, S, 1))            # [b,S,1]
        cols = (ox.view(b, 1, 1) + ar.view(1, 1, S))            # [b,1,S]
        bidx = torch.arange(b, device=self.device).view(b, 1, 1)
        out = xp.permute(0, 2, 3, 1)[bidx, rows, cols]          # [b,S,S,3]
        return out.permute(0, 3, 1, 2).contiguous()

    def _rand(self, b, lo, hi, gen, shape=None):
        return torch.empty(shape or (b, 1, 1, 1), device=self.device).uniform_(lo, hi,
                                                                              generator=gen)

    def _augment_strong(self, idx: torch.Tensor, gen: torch.Generator,
                        min_scale: float = 0.35, jitter: float = 0.4) -> torch.Tensor:
        """RandomResizedCrop + flip + colour jitter, batched, returning float in [0,1].

        Exists because every run in this benchmark drives training CE to ~0.00 -- flip plus a
        pad-4 crop does not regularise a 23M-parameter ResNet-50 on 9k-116k images at all. The
        whole comparison therefore sits in a memorising regime, and this is the arm that tests
        whether the dense-head advantage survives outside it.

        The crop is done with one affine_grid/grid_sample rather than a per-sample loop, so the
        cost is a single batched resample. The flip is folded into the affine matrix's x-scale.
        """
        import torch.nn.functional as F
        xb = self.x[idx].float().div_(255.0)                    # [b,3,S,S]
        b, S = xb.shape[0], self.size
        area = self._rand(b, min_scale, 1.0, gen, (b,))
        logr = self._rand(b, -0.2877, 0.2877, gen, (b,))        # log(3/4) .. log(4/3)
        ratio = torch.exp(logr)
        w = torch.sqrt(area * ratio).clamp(max=1.0)
        h = torch.sqrt(area / ratio).clamp(max=1.0)
        # centre must leave the crop inside the image: cx in [-(1-w), +(1-w)] in grid coords
        cx = (torch.rand(b, device=self.device, generator=gen) * 2 - 1) * (1 - w)
        cy = (torch.rand(b, device=self.device, generator=gen) * 2 - 1) * (1 - h)
        flip = torch.where(torch.rand(b, device=self.device, generator=gen) < 0.5,
                           -torch.ones(b, device=self.device), torch.ones(b, device=self.device))
        theta = torch.zeros(b, 2, 3, device=self.device)
        theta[:, 0, 0] = w * flip
        theta[:, 1, 1] = h
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, (b, 3, S, S), align_corners=False)
        xb = F.grid_sample(xb, grid, mode="bilinear", padding_mode="reflection",
                           align_corners=False)
        if jitter > 0:
            lo, hi = 1.0 - jitter, 1.0 + jitter
            xb = xb * self._rand(b, lo, hi, gen)                                  # brightness
            m = xb.mean(dim=(1, 2, 3), keepdim=True)
            xb = (xb - m) * self._rand(b, lo, hi, gen) + m                        # contrast
            lum = (xb * torch.tensor([0.299, 0.587, 0.114], device=self.device)
                   .view(1, 3, 1, 1)).sum(1, keepdim=True)
            xb = (xb - lum) * self._rand(b, lo, hi, gen) + lum                    # saturation
            xb = xb.clamp_(0.0, 1.0)
        return xb

    def normalise(self, xb: torch.Tensor) -> torch.Tensor:
        # uint8 from the weak paths, float in [0,1] from _augment_strong.
        xf = (xb.float().div_(255.0) if xb.dtype == torch.uint8 else xb)
        xf = (xf - self._mean) / self._std
        if self.channels_last:
            xf = xf.contiguous(memory_format=torch.channels_last)
        return xf

    # -------------------------------------------------------------- iteration
    def train_batches(self, batch_size: int, gen: torch.Generator, augment: str = "per_batch",
                      drop_last: bool = True, order: Optional[torch.Tensor] = None):
        perm = torch.randperm(self.n, device=self.device, generator=gen) if order is None else order
        nb = self.n // batch_size if drop_last else -(-self.n // batch_size)
        for i in range(nb):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            if idx.numel() == 0:
                continue
            xb = (self._augment_strong(idx, gen, self.rrc_min_scale, self.jitter_strength)
                  if augment == "strong" else self._augment(idx, augment, gen))
            yield self.normalise(xb), self.y[idx], idx

    def eval_batches(self, batch_size: int, idx: Optional[torch.Tensor] = None):
        n = self.n if idx is None else idx.numel()
        for i in range(0, n, batch_size):
            sl = (torch.arange(i, min(i + batch_size, n), device=self.device)
                  if idx is None else idx[i:i + batch_size])
            yield self.normalise(self.x[sl]), self.y[sl], sl


def probe_indices(y: torch.Tensor, num_classes: int, per_class: int,
                  gen: torch.Generator) -> torch.Tensor:
    """Up to ``per_class`` indices per class, for the end-of-epoch dense competence pass.

    Every class is represented (tail classes contribute all of their samples), so the resulting
    K x C matrix has no unmeasured cells and the imputation rule of Eq (3) becomes dead code.
    """
    keep = []
    for c in range(num_classes):
        ic = (y == c).nonzero(as_tuple=True)[0]
        if ic.numel() == 0:
            continue
        if ic.numel() > per_class:
            sel = torch.randperm(ic.numel(), device=ic.device, generator=gen)[:per_class]
            ic = ic[sel]
        keep.append(ic)
    return torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=y.device)


def group_stratified_order(y: torch.Tensor, assignment: torch.Tensor, batch_size: int,
                           num_experts: int, gen: torch.Generator) -> torch.Tensor:
    """Permutation whose consecutive ``batch_size`` blocks are balanced across assignment groups.

    Fixes the defect that at K=32 / batch 64 the mean group size is 2 and ``mask.sum() < 2``
    silently drops most groups -- which means a large share of few-shot samples never contribute
    to the backbone gradient at all. This changes only the sampling order, not the batch size or
    the learning rate, so it is a much smaller perturbation than raising the batch size.
    """
    gid = assignment[y]                                         # [N] group of each sample
    pools = []
    for g in range(num_experts):
        ig = (gid == g).nonzero(as_tuple=True)[0]
        if ig.numel() == 0:
            continue
        pools.append(ig[torch.randperm(ig.numel(), device=ig.device, generator=gen)])
    if not pools:
        return torch.randperm(y.numel(), device=y.device, generator=gen)
    G = len(pools)
    per = max(1, batch_size // G)
    cursors = [0] * G
    out, exhausted = [], 0
    while exhausted < G:
        exhausted = 0
        for j, pool in enumerate(pools):
            c = cursors[j]
            if c >= pool.numel():
                exhausted += 1
                continue
            out.append(pool[c:c + per])
            cursors[j] = c + per
    return torch.cat(out)
