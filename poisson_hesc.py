"""Ancestral samples from the normalized hierarchical model.

a1 ~ Poisson(mu1)                     [layer-2 "object" counts over G]
ahat0 = Phi1 a1  (+ eps)              [top-down prediction, nonneg]
a0 ~ Poisson(ahat0)                   [layer-1 "part" counts over G]
I  = Phi0 a0 + N(0, sigma^2)          [image]

Phi0, Phi1 are group (translation) convolutions with orbit-structured
template dictionaries.  Also samples the Gamma(1)=Exponential variant
a0 ~ Exp(mean=ahat0) for comparison.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

N, K0, K1 = 88, 11, 25
C0, C1 = 4, 3
THETAS = np.deg2rad([0, 45, 90, 135])
AMP, MU1, SIGMA, EPS = 1.6, 4.8e-4, 0.12, 1e-6

AMPLITUDE = 1.6
COEFFS_SIDE_HIGH = 11
COEFFS_SIDE_LOW = 25
IMAGE_SIDE = 88
NUM_FEATURES_HIGH = 3
NUM_FEATURES_LOW = 4
PRIOR_RATE_HIGH = 4.8e-4
PRIOR_RATE_LOW = 1e-6
STD_DEV = 0.12

def line_template(K, theta, s_long=2.6, s_wide=0.85):
    """Signed, bandpass, oriented line element; l2-normalized."""
    y, x = np.mgrid[0:K, 0:K] - (K - 1) / 2.0
    u = x * np.cos(theta) + y * np.sin(theta)      # along the line
    v = -x * np.sin(theta) + y * np.cos(theta)     # across the line
    g = np.exp(-u**2 / (2 * s_long**2) - v**2 / (2 * s_wide**2))
    g = g * (1.0 - (v / s_wide) ** 2)              # bandpass across
    g -= g.mean()
    return g / np.linalg.norm(g)


phi0 = np.stack([line_template(K0, t) for t in THETAS])       # (C0,K0,K0)

# layer-1 templates: NONNEGATIVE impulse maps over (channel, dy, dx)
def build_phi1():
    p = np.zeros((C1, C0, K1, K1))
    c = K1 // 2

    def put(k, ch, dy, dx):
        p[k, ch, c + dy, c + dx] = AMP

    # 0: box  (horizontal top/bottom, vertical left/right)
    for dx in (-5, 0, 5):
        put(0, 0, -8, dx); put(0, 0, 8, dx)
    for dy in (-5, 0, 5):
        put(0, 2, dy, -8); put(0, 2, dy, 8)
    # 1: L-corner
    for dx in (-7, -2, 3, 8):
        put(1, 0, 8, dx)
    for dy in (-7, -2, 3):
        put(1, 2, dy, -8)
    # 2: X  (two diagonals crossing)
    for d in (-7, -3, 0, 3, 7):
        put(2, 1, d, d); put(2, 3, -d, d)
    return p


phi1 = build_phi1()


def scatter(canvas, tmpl, y, x, n):
    """Add n*tmpl centered at (y,x) into canvas, clipping at borders."""
    kh, kw = tmpl.shape[-2:]
    y0, x0 = y - kh // 2, x - kw // 2
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(canvas.shape[-2], y0 + kh), min(canvas.shape[-1], x0 + kw)
    if ys >= ye or xs >= xe:
        return
    canvas[..., ys:ye, xs:xe] += n * tmpl[..., ys - y0:ye - y0, xs - x0:xe - x0]


def Phi1(a1):
    """(C1,N,N) counts -> (C0,N,N) nonneg prediction."""
    out = np.zeros((C0, N, N))
    for k, y, x in zip(*np.nonzero(a1)):
        scatter(out, phi1[k], y, x, a1[k, y, x])
    return out


def Phi0(a0):
    """(C0,N,N) coefficients -> (N,N) image."""
    out = np.zeros((N, N))
    for c, y, x in zip(*np.nonzero(a0)):
        scatter(out, phi0[c], y, x, a0[c, y, x])
    return out


def sample():
    m = K1 // 2 + 2
    a1 = np.zeros((C1, N, N), dtype=int)
    a1[:, m:N - m, m:N - m] = rng.poisson(MU1, (C1, N - 2 * m, N - 2 * m))
    ahat0 = Phi1(a1) + EPS
    a0_pois = rng.poisson(ahat0)
    a0_exp = np.where(ahat0 > 1e-3, rng.exponential(np.maximum(ahat0, 1e-12)), 0.0)
    I_p = Phi0(a0_pois) + SIGMA * rng.standard_normal((N, N))
    I_e = Phi0(a0_exp) + SIGMA * rng.standard_normal((N, N))
    return a1, ahat0, a0_pois, a0_exp, I_p, I_e


# ---------------- figure 1: the dictionary ----------------
fig, ax = plt.subplots(1, C0 + C1, figsize=(2.0 * (C0 + C1), 2.4))
for c in range(C0):
    ax[c].imshow(phi0[c], cmap="gray")
    ax[c].set_title(f"$\\phi_0^{c+1}$  {int(np.rad2deg(THETAS[c]))}°", fontsize=10)
names = ["box", "L-corner", "X"]
for k in range(C1):
    a = np.zeros((C0, N, N)); a[:, N//2-K1//2:N//2+K1//2+1, N//2-K1//2:N//2+K1//2+1] = phi1[k]
    ax[C0 + k].imshow(Phi0(a)[N//2-20:N//2+20, N//2-20:N//2+20], cmap="gray")
    ax[C0 + k].set_title(f"$\\Phi_0\\phi_1^{k+1}$  ({names[k]})", fontsize=10)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Layer-0 templates (signed) and layer-1 templates back-projected into image space",
             fontsize=11)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/dictionary.png", dpi=150, bbox_inches="tight")

# ---------------- figure 2: ancestral samples ----------------
NS = 4
fig, ax = plt.subplots(NS, 5, figsize=(15.5, 3.1 * NS))
cols = ["$a_1$ support (layer-2 counts)", "$\\hat a_0 = \\Phi_1 a_1$",
        "$a_0 \\sim$ Poisson$(\\hat a_0)$", "$I$  (Poisson $a_0$)",
        "$I$  (Gamma(1) $a_0$)"]
marks, colr = ["s", "^", "x"], ["tab:red", "tab:blue", "tab:green"]
for r in range(NS):
    a1, ahat0, a0p, a0e, Ip, Ie = sample()
    ax[r, 0].set_facecolor("#f7f7f7")
    for k, y, x in zip(*np.nonzero(a1)):
        ax[r, 0].scatter(x, y, marker=marks[k], c=colr[k], s=60 * a1[k, y, x])
    ax[r, 0].set_xlim(0, N); ax[r, 0].set_ylim(N, 0); ax[r, 0].set_aspect("equal")
    ax[r, 1].imshow(ahat0.sum(0), cmap="magma")
    ax[r, 2].imshow(a0p.sum(0), cmap="magma")
    v = np.abs(Ip).max()
    ax[r, 3].imshow(Ip, cmap="gray", vmin=-v, vmax=v)
    v = np.abs(Ie).max()
    ax[r, 4].imshow(Ie, cmap="gray", vmin=-v, vmax=v)
    ax[r, 0].set_ylabel(f"sample {r+1}", fontsize=11)
    print(f"sample {r+1}: objects={a1.sum():2d}  parts predicted={int(round(ahat0.sum()/AMP)):3d}"
          f"  parts drawn={a0p.sum():3d}  ||a0||_0={np.count_nonzero(a0p):3d}"
          f"  ||ahat0||_1={ahat0.sum():7.2f}")
    for c in range(5):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(cols[c], fontsize=11)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/samples.png", dpi=150, bbox_inches="tight")

# ---------------- equivariance check ----------------
a1, ahat0, a0p, _, _, _ = sample()
dy, dx = 9, -13
a0s = np.roll(a0p, (dy, dx), axis=(1, 2))
lhs, rhs = Phi0(a0s), np.roll(Phi0(a0p), (dy, dx), axis=(0, 1))
inner = (slice(20, N - 20), slice(20, N - 20))
print(f"\nequivariance  max|Phi0(L_h a0) - L_h Phi0(a0)| = {np.abs(lhs[inner]-rhs[inner]).max():.3e}"
      f"   (image scale {np.abs(rhs).max():.2f})")
