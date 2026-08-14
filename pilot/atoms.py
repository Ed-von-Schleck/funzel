"""The unified metamorphic atom (plans/metamorphic-splats.md §4), batched, in PyTorch.

    f(x) = [sum_a c_a H_a(u)] cos(w·u) + [sum_a d_a H_a(u)] sin(w·u),
    all x exp(-0.5 ||u||^beta),   u = D^{-1} R^T (x - mu),  D = diag(s1, s2).

Conventions:
  - Coordinates are pixels. Rendering supersamples ss x ss per pixel and box-averages,
    which stands in for the pixel filter uniformly across all atom types (the plan's
    analytic EWA filter for beta=2 is a later optimization; target and model share the
    renderer, so the sanity phase's landscape questions are unaffected).
  - The carrier w lives in the atom frame; the image-space frequency is k = R (w / s).
  - The sin block and d-coefficients exist only when the carrier is unlocked (the
    omega-conditional accounting of §4); a blunt atom is exactly a Gaussian.
  - beta is parametrized as beta = 1 + 7*sigmoid(raw), range (1, 8), blunt at raw
    such that beta = 2.
  - Everything is batched over a leading seed dimension B: independent single-atom
    fits share one optimizer (losses sum, gradients decouple).
"""

import math
import torch


def rotmats(rot):
    c, s = torch.cos(rot), torch.sin(rot)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)


def beta_raw_init(B):
    # raw value at which 1 + 7*sigmoid(raw) == 2
    return torch.full((B,), math.log(1.0 / 6.0))


def atom_terms(geom, H, W, order, with_sin, ss=2):
    """Return the linear-term images (B, n_terms, H, W) for the current geometry.

    n_terms = n_basis (cos block) [+ n_basis (sin block) if with_sin], where
    n_basis = 1 (order 0) or 3 (order 1: 1, u1, u2). The rendered atom is the
    coefficient-weighted sum of these terms, so least squares over coefficients
    and the fit loop share one code path.
    """
    mu, log_s, rot = geom["mu"], geom["log_s"], geom["rot"]
    B = mu.shape[0]
    dev = mu.device
    ys = (torch.arange(H * ss, device=dev, dtype=torch.float32) + 0.5) / ss - 0.5
    xs = (torch.arange(W * ss, device=dev, dtype=torch.float32) + 0.5) / ss - 0.5
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    P = torch.stack([X, Y], -1)  # (Hs, Ws, 2)
    d = P.unsqueeze(0) - mu[:, None, None, :]  # (B,Hs,Ws,2)
    R = rotmats(rot)
    u = torch.einsum("bji,bhwj->bhwi", R, d) / torch.exp(log_s)[:, None, None, :]
    r2 = (u * u).sum(-1).clamp_min(1e-8)

    if "beta_raw" in geom:
        beta = 1.0 + 7.0 * torch.sigmoid(geom["beta_raw"])
        env = torch.exp(-0.5 * r2 ** (beta[:, None, None] / 2.0))
    else:
        env = torch.exp(-0.5 * r2)

    basis = [torch.ones_like(r2)]
    if order >= 1:
        basis += [u[..., 0], u[..., 1]]

    terms = []
    if with_sin:
        w = geom["omega"]
        phase = (u * w[:, None, None, :]).sum(-1)
        cosp, sinp = torch.cos(phase), torch.sin(phase)
        terms += [b * cosp * env for b in basis]
        terms += [b * sinp * env for b in basis]
    else:
        terms += [b * env for b in basis]

    T = torch.stack(terms, 1)  # (B, n_terms, Hs, Ws)
    T = T.reshape(B, T.shape[1], H, ss, W, ss).mean((3, 5))
    return T


def render(atom, H, W, ss=2):
    with_sin = "omega" in atom
    order = 1 if atom["n_basis"] == 3 else 0
    T = atom_terms(atom, H, W, order, with_sin, ss)
    if "phi" in atom:  # polar appearance: c = r cos(phi), d = r sin(phi), per component
        c = atom["r"] * torch.cos(atom["phi"])
        d = atom["r"] * torch.sin(atom["phi"])
        coeffs = torch.cat([c, d], 1)
    elif with_sin:
        coeffs = torch.cat([atom["c"], atom["d"]], 1)
    else:
        coeffs = atom["c"]
    return torch.einsum("bn,bnhw->bhw", coeffs, T)


def ls_coeffs(geom, target, order, with_sin, ss=2):
    """Closed-form amplitude solve at birth (plan §5): least squares of the target
    against the atom's linear terms, geometry frozen. Returns (B, n_terms)."""
    with torch.no_grad():
        B, H, W = target.shape
        T = atom_terms(geom, H, W, order, with_sin, ss)  # (B,n,H,W)
        A = T.reshape(B, T.shape[1], -1).transpose(1, 2)  # (B, HW, n)
        y = target.reshape(B, -1, 1)
        sol = torch.linalg.lstsq(A, y).solution.squeeze(-1)  # (B, n)
    return sol


def gaussian_blur(img, sigma=3.0):
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    B = img.shape[0]
    v = torch.nn.functional.conv2d(img.unsqueeze(1), k.view(1, 1, -1, 1), padding=(r, 0))
    v = torch.nn.functional.conv2d(v, k.view(1, 1, 1, -1), padding=(0, r))
    return v.squeeze(1)


def moments_init(target, jitter=None, gen=None):
    """Shared shape-init rule: position at the peak of the smoothed |target| (plus
    optional uniform jitter, modeling birth imprecision), isotropic scale from the
    second moments of the smoothed |target|."""
    B, H, W = target.shape
    w = gaussian_blur(target.abs(), 3.0).clamp_min(0)
    flat = w.reshape(B, -1)
    idx = flat.argmax(1)
    py = (idx // W).float()
    px = (idx % W).float()
    pos = torch.stack([px, py], -1)
    if jitter is not None:
        pos = pos + (torch.rand((B, 2), generator=gen) * 2 - 1) * jitter
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    tot = flat.sum(1)
    mx = (w * X).sum((1, 2)) / tot
    my = (w * Y).sum((1, 2)) / tot
    vx = (w * (X - mx[:, None, None]) ** 2).sum((1, 2)) / tot
    vy = (w * (Y - my[:, None, None]) ** 2).sum((1, 2)) / tot
    sigma = ((vx + vy) / 2).sqrt().clamp(2.0, 12.0)
    return pos, sigma


def spectrogram_birth(target, pos, sigma, k_min=0.12 * math.pi, patch=32, gen=None):
    """Measure the carrier at birth: Hann-windowed FFT of the patch around pos,
    peak |F| outside the separation disc ||k|| >= max(k_min, 1/sigma). Returns the
    image-space carrier k (B, 2)."""
    B, H, W = target.shape
    half = patch // 2
    win1 = torch.hann_window(patch, periodic=False)
    win = win1[:, None] * win1[None, :]
    fy = torch.fft.fftfreq(patch) * 2 * math.pi
    fx = torch.fft.fftfreq(patch) * 2 * math.pi
    KY, KX = torch.meshgrid(fy, fx, indexing="ij")
    kmag = (KX ** 2 + KY ** 2).sqrt()
    ks = []
    for b in range(B):
        cx = int(pos[b, 0].round().clamp(half, W - half - 1))
        cy = int(pos[b, 1].round().clamp(half, H - half - 1))
        p = target[b, cy - half : cy + half, cx - half : cx + half] * win
        F = torch.fft.fft2(p).abs()
        sep = torch.maximum(torch.tensor(k_min), 1.0 / sigma[b])
        F = torch.where(kmag >= sep, F, torch.zeros_like(F))
        i = int(F.reshape(-1).argmax())
        ks.append(torch.tensor([KX.reshape(-1)[i], KY.reshape(-1)[i]]))
    return torch.stack(ks)


def candidate_birth(atom, target, order, with_sin, ss=2, r_frac=0.6, n_ang=8,
                    rots=(0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)):
    """Matched-filter birth over a joint (position, rotation) candidate set, selected
    by closed-form LS residual.

    Rationale: the order-1 basis is proportional to dG/dmu, so an order-1 atom has a
    Taylor-absorbed twin -- a plain Gaussian shifted onto its dominant lobe reproduces
    it to first order -- and the twin is a genuine secondary basin whose catchment
    contains the |target| peak, i.e. the natural spawn point. Selecting the birth pose
    over candidates escapes the twin in most cases (17/20 vs 12/20 single-candidate on
    the sanity cell); the remainder are left to population dynamics, which see the
    twin's ~9% leftover residual. See pilot/results/sanity_e2_diagnosis.txt."""
    with torch.no_grad():
        B, H, W = target.shape
        sigma = torch.exp(atom["log_s"][:, 0])
        offs = [torch.zeros(B, 2)]
        for a in range(n_ang):
            ang = 2 * math.pi * a / n_ang
            offs.append(r_frac * sigma[:, None] *
                        torch.tensor([math.cos(ang), math.sin(ang)]))
        best = None
        for off in offs:
            for r in rots:
                geom = {"mu": atom["mu"].detach() + off,
                        "log_s": atom["log_s"].detach(),
                        "rot": torch.full((B,), r)}
                if "beta_raw" in atom:
                    geom["beta_raw"] = atom["beta_raw"].detach()
                if with_sin:
                    geom["omega"] = atom["omega"].detach()
                sol = ls_coeffs(geom, target, order, with_sin, ss)
                T = atom_terms(geom, H, W, order, with_sin, ss)
                res = ((torch.einsum("bn,bnhw->bhw", sol, T) - target) ** 2).sum((1, 2))
                if best is None:
                    best = {"res": res, "mu": geom["mu"], "rot": geom["rot"]}
                else:
                    m = res < best["res"]
                    best["res"] = torch.where(m, res, best["res"])
                    best["mu"] = torch.where(m[:, None], geom["mu"], best["mu"])
                    best["rot"] = torch.where(m, geom["rot"], best["rot"])
        atom["mu"] = best["mu"].clone().requires_grad_()
        atom["rot"] = best["rot"].clone().requires_grad_()
    return atom


def image_freq(atom):
    """k = R (w / s), the image-space carrier of a fitted atom."""
    with torch.no_grad():
        R = rotmats(atom["rot"])
        w = atom["omega"] / torch.exp(atom["log_s"])
        return torch.einsum("bij,bj->bi", R, w)


def atom_frame_omega(k, log_s, rot):
    """Inverse of image_freq at given geometry: w = s * (R^T k)."""
    R = rotmats(rot)
    return torch.einsum("bji,bj->bi", R, k) * torch.exp(log_s)


LRS = {"mu": 0.15, "log_s": 0.02, "rot": 0.02, "omega": 0.03,
       "c": 0.02, "d": 0.02, "r": 0.02, "phi": 0.05, "beta_raw": 0.05}


def fit(atom, target, iters=3000, ss=2, lrs=LRS):
    """Adam with per-group LRs; two step-decays; batched over seeds (losses sum,
    so the B fits are independent)."""
    groups = [{"params": [v], "lr": lrs[k]} for k, v in atom.items()
              if torch.is_tensor(v) and v.requires_grad]
    opt = torch.optim.Adam(groups)
    B, H, W = target.shape
    for i in range(iters):
        if i in (int(iters * 0.6), int(iters * 0.85)):
            for g in opt.param_groups:
                g["lr"] *= 0.5
        opt.zero_grad(set_to_none=True)
        img = render(atom, H, W, ss)
        loss = ((img - target) ** 2).sum(dim=(1, 2))  # per-seed, summed below
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        img = render(atom, H, W, ss)
        rel = ((img - target) ** 2).sum((1, 2)).sqrt() / (target ** 2).sum((1, 2)).sqrt()
    return rel
