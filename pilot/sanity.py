"""Sanity phase of plans/metamorphic-splats.md: planted-atom recovery, one test per knob.

Each cell plants ONE atom of the unified family on a 64x64 image and asks whether a
single atom, initialized by a given arm's own birth policy, recovers it by Adam descent.
This validates the mechanisms (and the harness) before any GPU-days are spent. Recovery
means relative L2 error < 0.05 at the end of the fit.

PRE-REGISTERED EXPECTATIONS (fixed before the first run):
  E1  gauss    (blunt atom, random position init, large init scale)   >= 18/20
  E2  hermite  (order-1 unlocked, LS coefficients at birth)           >= 18/20
  E3  beta     (beta unlocked from 2, planted beta=5)                 >= 16/20
  E4  gabor + spectrogram birth (A3/A2a policy), both frequencies     >= 16/20 each
  E5  gabor + random-omega birth (A2b policy) at k=0.6pi: recovery rate at least
      50 points below E4's at the same frequency — the landscape claim: richness
      with uninformed nonlinear init should fail where informed birth succeeds
  E6  gabor + eps-omega (near-zero carrier, descent must find the frequency)
      at k=0.6pi fails (<= 4/20) — the omega~0 saddle: without a birth measurement
      or an annealing schedule, descent cannot climb to a distant carrier
  E7  polar parametrization (A4d policy: same LS birth, magnitudes+phases descended)
      at k=0.6pi recovers at most as often as Cartesian (E4); the gap is the
      quadrature effect previewed at n=1 and is reported, not thresholded

A failure of E1-E3 is a harness bug until proven otherwise (instrument before
interpretation); E4-E7 are the readings. One atom is not the population dynamics of
the real experiment — capture radius, birth accuracy, and parametrization geometry
are what transfer; absolute rates do not.

Runtime: ~11 cells x 20 seeds, batched; minutes on CPU. Output: pilot/results/sanity.txt
"""

import math
import sys
import time

import torch

from atoms import (LRS, atom_frame_omega, beta_raw_init, candidate_birth, fit,
                   image_freq, ls_coeffs, moments_init, render, spectrogram_birth)

H = W = 64
B = 20  # seeds per cell
IT = 3000
CTR = torch.tensor([31.5, 31.5])


def g(seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def plant_gauss(gen):
    rot = torch.rand((B,), generator=gen) * math.pi
    return {"mu": CTR.repeat(B, 1), "log_s": torch.log(torch.tensor([7.0, 3.0])).repeat(B, 1),
            "rot": rot, "c": torch.tensor([[1.0]]).repeat(B, 1), "n_basis": 1}


def plant_hermite(gen):
    a = plant_gauss(gen)
    a["c"] = torch.tensor([[0.8, 0.5, -0.4]]).repeat(B, 1)
    a["n_basis"] = 3
    return a


def plant_beta(gen):
    raw = math.log(4.0 / 3.0)  # beta = 1 + 7*sigmoid(raw) = 5
    return {"mu": CTR.repeat(B, 1), "log_s": torch.log(torch.tensor([6.0, 6.0])).repeat(B, 1),
            "rot": torch.zeros(B), "c": torch.tensor([[1.0]]).repeat(B, 1),
            "beta_raw": torch.full((B,), raw), "n_basis": 1}


def plant_gabor(gen, kmag):
    ang = torch.rand((B,), generator=gen) * math.pi
    k = torch.stack([kmag * torch.cos(ang), kmag * torch.sin(ang)], -1)
    log_s = torch.log(torch.tensor([6.0, 6.0])).repeat(B, 1)
    rot = torch.zeros(B)
    return {"mu": CTR.repeat(B, 1), "log_s": log_s, "rot": rot,
            "c": torch.tensor([[1.0]]).repeat(B, 1), "d": torch.tensor([[0.3]]).repeat(B, 1),
            "omega": atom_frame_omega(k, log_s, rot), "n_basis": 1}, k


def geom_init(target, gen, jitter=3.0):
    pos, sigma = moments_init(target, jitter=jitter, gen=gen)
    return {"mu": pos.clone().requires_grad_(),
            "log_s": torch.log(torch.stack([sigma, sigma], -1)).requires_grad_(),
            "rot": torch.zeros(B, requires_grad=True)}


def finish(atom, target, order, with_sin, polar=False):
    """LS coefficients at birth, then Cartesian or polar appearance parameters."""
    sol = ls_coeffs({k: v.detach() if torch.is_tensor(v) else v for k, v in atom.items()},
                    target, order, with_sin)
    n = 3 if order else 1
    if not with_sin:
        atom["c"] = sol.clone().requires_grad_()
    elif polar:
        c, d = sol[:, :n], sol[:, n:]
        atom["r"] = (c ** 2 + d ** 2).sqrt().requires_grad_()
        atom["phi"] = torch.atan2(d, c).requires_grad_()
    else:
        atom["c"] = sol[:, :n].clone().requires_grad_()
        atom["d"] = sol[:, n:].clone().requires_grad_()
    atom["n_basis"] = n
    return atom


def k_err(atom, k_true):
    kf = image_freq(atom)
    e1 = (kf - k_true).norm(dim=-1)
    e2 = (kf + k_true).norm(dim=-1)  # carrier sign symmetry
    return torch.minimum(e1, e2) / k_true.norm(dim=-1)


def run():
    rows = []
    t00 = time.time()

    def report(name, rel, extra=""):
        rec = int((rel < 0.05).sum())
        rows.append(f"{name:34s} {rec:2d}/20 recovered   median rel_err {rel.median():.4f}{extra}")
        print(rows[-1], flush=True)

    # E1: blunt Gaussian, random position (capture-radius test), large init scale
    gen = g(1)
    with torch.no_grad():
        tgt = render(plant_gauss(gen), H, W)
    pos = torch.rand((B, 2), generator=gen) * (0.8 * W) + 0.1 * W
    atom = {"mu": pos.requires_grad_(),
            "log_s": torch.log(torch.full((B, 2), 12.0)).requires_grad_(),
            "rot": torch.zeros(B, requires_grad=True)}
    atom = finish(atom, tgt, 0, False)
    report("E1 gauss / random-pos", fit(atom, tgt, IT))

    # E2: order-1 Hermite. Birth policy is candidate-set LS (see atoms.candidate_birth
    # and results/sanity_e2_diagnosis.txt): single-candidate birth at the |target|
    # peak lands in the Taylor-twin basin 8/20 times. The pre-registered floor
    # (>=18/20) is unchanged; the cell reports against it honestly.
    gen = g(2)
    with torch.no_grad():
        tgt = render(plant_hermite(gen), H, W)
    atom = candidate_birth(geom_init(tgt, gen), tgt, 1, False)
    atom = finish(atom, tgt, 1, False)
    report("E2 hermite / candidate-birth", fit(atom, tgt, IT))

    # E3: beta=5 plant, beta unlocked from 2
    gen = g(3)
    with torch.no_grad():
        tgt = render(plant_beta(gen), H, W)
    atom = geom_init(tgt, gen)
    atom["beta_raw"] = beta_raw_init(B).requires_grad_()
    atom = finish(atom, tgt, 0, False)
    report("E3 beta / unlocked-from-2", fit(atom, tgt, IT))

    # E4-E7: gabor cells at two frequencies x four policies.
    # Seeds via crc32, NOT hash(): Python salts str hashes per process, which made
    # the random-omega cells irreproducible across runs.
    import zlib
    for kmag, ktag in [(0.25 * math.pi, "0.25pi"), (0.60 * math.pi, "0.60pi")]:
        for policy in ["spectro", "random", "eps", "polar"]:
            gen = g(zlib.crc32(f"{ktag}/{policy}".encode()))
            planted, k_true = plant_gabor(gen, kmag)
            with torch.no_grad():
                tgt = render(planted, H, W)
            atom = geom_init(tgt, gen)
            with torch.no_grad():
                sigma = torch.exp(atom["log_s"][:, 0])
                if policy in ("spectro", "polar"):
                    k0 = spectrogram_birth(tgt, atom["mu"], sigma, gen=gen)
                elif policy == "random":
                    mag = (0.1 + 0.7 * torch.rand((B,), generator=gen)) * math.pi
                    ang = torch.rand((B,), generator=gen) * 2 * math.pi
                    k0 = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], -1)
                else:  # eps: near-zero carrier, descent must find the frequency
                    ang = torch.rand((B,), generator=gen) * 2 * math.pi
                    k0 = 0.05 * math.pi * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
                w0 = atom_frame_omega(k0, atom["log_s"].detach(), atom["rot"].detach())
            atom["omega"] = w0.clone().requires_grad_()
            atom = finish(atom, tgt, 0, True, polar=(policy == "polar"))
            rel = fit(atom, tgt, IT)
            report(f"E4-7 gabor k={ktag} / {policy}", rel,
                   f"   median k_err {k_err(atom, k_true).median():.3f}")

    rows.append(f"total {time.time() - t00:.0f}s")
    with open("results/sanity.txt", "w") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()
