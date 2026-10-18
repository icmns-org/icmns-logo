"""ICMNS logo generator: lateral view of a convoluted brain, drawn as fibres.

The mathematics:
  * gyri and sulci are a Turing pattern: a stationary solution of the
    Swift-Hohenberg equation  u_t = r u - (1 + Lap/k0^2)^2 u - u^3
    on the cortical domain (u pinned negative outside it, so folds run
    along the outline and the lateral fissure);
  * each gyrus is drawn as several nested level sets {u = c}, coloured by
    local fibre orientation as in diffusion-tensor tractography;
  * cerebellar folia are level sets of a rippled ellipse function;
  * the brainstem is a bundle of parallel offsets of an integral sign.
Text is converted to outlines, so the SVG needs no fonts.
Run:  python3 make_logo.py   (needs numpy + matplotlib)
"""
import argparse
import numpy as np
import contourpy
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


W, H = 1000, 1080                     # SVG canvas
CX, CY, S = 500, 400, 345             # brain origin and scale (px per unit)
# colour roles: AP = anterior-posterior fibres, SI = superior-inferior,
# OB1/OB2 = the two oblique directions (as in DTI colour coding)
PALETTES = {
    "warm": dict(BG="#1e1c1d", AP="#ef6b5f", SI="#9fe07c", OB1="#e7ef7a", OB2="#3f9bd6",
                 TITLE="#f2f0ec", SUB="#bdb8b0"),
    "cool": dict(BG="#0b1021", AP="#4c8dff", SI="#3fe0c5", OB1="#8fd8ff", OB2="#9b7bff",
                 TITLE="#eef4ff", SUB="#9fb0cc"),
    # matched to the icmns.org "slate blue" site palette (light and dark mode);
    # AP is the site's link colour
    "site-light": dict(BG="#ffffff", AP="#1f5f8b", SI="#17766d", OB1="#2c5fa6", OB2="#4a3f9e",
                       TITLE="#1d2330", SUB="#5b6475"),
    "site-dark": dict(BG="#12161d", AP="#8cc8f0", SI="#6fe0cf", OB1="#b9e0ff", OB2="#b3a6ff",
                      TITLE="#e6e8ec", SUB="#9aa3b2"),
    # Nord (nordtheme.com): Frost blues and Aurora purple. Dark mode uses the Nord
    # colours as they are, on Polar Night; light mode darkens them slightly so the
    # fibres keep their contrast on Snow Storm
    "nord-light": dict(BG="#eceff4", AP="#5e81ac", SI="#5f9a98", OB1="#4f6f96", OB2="#8e6a8a",
                       TITLE="#2e3440", SUB="#4c566a"),
    "nord-dark": dict(BG="#2e3440", AP="#81a1c1", SI="#8fbcbb", OB1="#88c0d0", OB2="#b48ead",
                      TITLE="#eceff4", SUB="#d8dee9"),
}
args = argparse.ArgumentParser(description=__doc__.split("\n")[0])
args.add_argument("--palette", default="cool", choices=PALETTES)
args.add_argument("--mask", help="image whose shape gives the cortex outline "
                  "(PNG with transparency, or a dark/light silhouette)")
args.add_argument("--seed", type=int, default=3)
args.add_argument("--no-cerebellum", action="store_true",
                  help="omit the cerebellum (e.g. if the mask image already includes it)")
args.add_argument("--stroke-scale", type=float, default=1.0,
                  help="multiply all fibre widths (thicker lines read better on light backgrounds)")
args.add_argument("--outline", default="classic", choices=["classic", "anatomical"],
                  help="cortex outline: the original ellipses, or a lateral-view hemisphere")
args.add_argument("--stem", default="classic", choices=["classic", "anatomical"],
                  help="integral-sign brainstem: original placement, or emerging under the "
                  "cerebrum in front of the cerebellum, with a pons")
args.add_argument("--cerebellum", default="folia", choices=["folia", "phase"],
                  help="folia in an oval (original), or a lateral-view cerebellum filled "
                  "with the FitzHugh-Nagumo flow and its limit cycle")
args.add_argument("--shift-back", type=float, default=0.0,
                  help="move the phase-portrait cerebellum and the anatomical brainstem this "
                  "far towards the back of the brain (brain units; the cerebrum is ~2.3 long)")
args.add_argument("--cerebellum-back", type=float, default=0.0,
                  help="move only the phase-portrait cerebellum further back (added to --shift-back)")
args.add_argument("--out", default="icmns")
args = args.parse_args()
P = PALETTES[args.palette]
rng = np.random.default_rng(args.seed)
BG, AP, SI, OB1, OB2 = P["BG"], P["AP"], P["SI"], P["OB1"], P["OB2"]

# ---------------------------------------------------------------- domain
nx, ny = 640, 480
x = np.linspace(-1.35, 1.35, nx)
y = np.linspace(-0.95, 1.07, ny)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)


def smin(a, b, k=30.0):
    return -np.log(np.exp(-k * a) + np.exp(-k * b)) / k


def smax(a, b, k=30.0):
    return -smin(-a, -b, k)


def seg_dist(X, Y, p, q):
    p, q = np.asarray(p), np.asarray(q)
    d = q - p
    t = np.clip(((X - p[0]) * d[0] + (Y - p[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(X - p[0] - t * d[0], Y - p[1] - t * d[1]), t


def cortex_sdf(X, Y):
    """Approximate signed 'distance' (<0 inside) of the cerebral cortex."""
    # hemisphere: ellipse, flattened underneath (orbital surface / occipital base)
    e = np.sqrt(((X - 0.0) / 1.16) ** 2 + ((Y - 0.12) / 0.80) ** 2) - 1.0
    floor = (-0.22 + 0.13 * X) - Y                      # cut below this line
    upper = smax(e * 0.8, floor)
    # temporal lobe: tilted ellipse, pole at the front
    c, s = np.cos(-0.18), np.sin(-0.18)
    u, v = X - 0.02, Y + 0.36
    u, v = c * u + s * v, -s * u + c * v
    temporal = np.sqrt((u / 0.74) ** 2 + (v / 0.27) ** 2) - 1.0
    brain = smin(upper, temporal * 0.3, k=40)
    # lateral (Sylvian) fissure: a thin cut from the temporal pole backwards
    d, t = seg_dist(X, Y, (0.75, -0.2), (-0.12, -0.02))
    gap = d - (0.012 + 0.018 * t)
    return smax(brain, -gap, k=60)


def closed_spline(P, n=40):
    """Closed Catmull-Rom spline through the points P."""
    P = np.asarray(P, float)
    Q = np.vstack([P[-1], P, P[0], P[1]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1:i + 3]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def polygon_sdf(X, Y, C):
    """Signed distance (<0 inside) from grid points to the closed polyline C."""
    from matplotlib.path import Path
    pts = np.c_[X.ravel(), Y.ravel()]
    d = np.empty(len(pts))
    for i in range(0, len(pts), 20000):            # distance to the outline, in chunks
        q = pts[i:i + 20000]
        d[i:i + 20000] = np.sqrt(((q[:, None, :] - C[None, :, :]) ** 2).sum(-1)).min(1)
    ins = Path(C).contains_points(pts)
    return np.where(ins, -d, d).reshape(X.shape)


def anatomical_sdf(X, Y):
    """Signed distance (<0 inside) to a left-hemisphere outline in lateral view,
    front to the right: frontal lobe, parietal arch, occipital pole, temporal
    lobe and pole, and a thin lateral (Sylvian) fissure."""
    P = [(1.10, 0.12), (1.02, 0.44), (0.82, 0.70), (0.50, 0.87), (0.12, 0.94),
         (-0.28, 0.91), (-0.62, 0.79), (-0.90, 0.57), (-1.07, 0.30), (-1.16, 0.03),
         (-1.12, -0.16), (-0.96, -0.27), (-0.70, -0.31), (-0.44, -0.37),
         (-0.16, -0.50), (0.14, -0.57), (0.42, -0.55), (0.64, -0.44), (0.74, -0.30),
         (0.75, -0.21), (0.86, -0.17), (1.02, -0.08)]
    sd = polygon_sdf(X, Y, closed_spline(P))
    # lateral fissure: from the temporal pole, backwards and slightly up
    g, t = seg_dist(X, Y, (0.74, -0.23), (-0.10, 0.02))
    gap = g - (0.006 + 0.010 * (1 - t))
    return smax(sd, -gap, k=80)


def image_sdf(path):
    """Cortex outline from an image: the shape is scaled to the same box as the
    built-in outline, and a blurred indicator stands in for a signed distance."""
    import matplotlib.image as mpimg
    img = mpimg.imread(path).astype(float)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 4 and img[..., 3].min() < 0.5:
        shape = img[..., 3] > 0.5                      # transparency
    else:
        lum = img[..., :3].mean(axis=2)
        bgl = np.median(np.r_[lum[0], lum[-1], lum[:, 0], lum[:, -1]])
        shape = np.abs(lum - bgl) > 0.25               # differs from the border
    r, c = np.nonzero(shape)
    shape = shape[r.min():r.max() + 1, c.min():c.max() + 1][::-1]  # y up
    h, w = shape.shape
    half_w = 1.18
    half_h = half_w * h / w
    x0, y0 = 0.0, 0.12 - (half_h - 0.80)              # keep the top at y = 0.92
    ii = np.clip(((Y - (y0 - half_h)) / (2 * half_h) * h).astype(int), -1, h)
    jj = np.clip(((X - (x0 - half_w)) / (2 * half_w) * w).astype(int), -1, w)
    valid = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    ind = np.zeros_like(X)
    ind[valid] = shape[ii[valid], jj[valid]]
    # Gaussian blur in Fourier space, then map 0.5 -> 0 (the boundary)
    sig = 0.02
    kx_ = 2 * np.pi * np.fft.fftfreq(X.shape[1], dx)
    ky_ = 2 * np.pi * np.fft.fftfreq(X.shape[0], dx)
    G = np.exp(-0.5 * sig ** 2 * (kx_[None, :] ** 2 + ky_[:, None] ** 2))
    blur = np.fft.ifft2(np.fft.fft2(ind) * G).real
    return (0.5 - blur) * 2 * sig * 1.25               # ~ signed distance near the edge


SD = (image_sdf(args.mask) if args.mask
      else anatomical_sdf(X, Y) if args.outline == "anatomical" else cortex_sdf(X, Y))


def sd_at(px, py):
    """SD at arbitrary points (nearest grid node)."""
    j = np.clip(np.rint((px - x[0]) / dx).astype(int), 0, nx - 1)
    i = np.clip(np.rint((py - y[0]) / (y[1] - y[0])).astype(int), 0, ny - 1)
    return SD[i, j]
inside = SD < 0

# ---------------------------------------- Swift-Hohenberg on the cortex
k0 = 2 * np.pi / 0.19                 # gyral wavelength (brain units)
kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
K2 = kx[None, :] ** 2 + ky[:, None] ** 2
Lop = 0.3 - (1 - K2 / k0 ** 2) ** 2   # linear operator in Fourier space
dt = 0.5
# start from noise; a short run of the SH flow gives a labyrinth (gyri),
# and pinning u < 0 just outside the cortex rounds the gyri off at the edge
u = 0.3 * rng.standard_normal((ny, nx))
pin = np.clip(-SD / 0.02, 0, 1)
n_it = 170
for it in range(n_it):
    nl = -u ** 3
    u = np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(nl)) / (1 - dt * Lop)).real
    if it > n_it - 60:                 # pin only at the end: keep the labyrinth
        u = pin * u + (1 - pin) * (-0.5)

# ---------------------------------------------------------------- helpers
fibres = []  # (layer, d, colour, width, opacity)


def to_d(seg, step=1):
    seg = seg[::step] if len(seg) > 3 * step else seg
    return "M" + " ".join(f"{CX + S * a:.1f},{CY - S * b:.1f}" for a, b in seg)


def orient_colour(seg):
    """DTI-style colour from the mean fibre direction (lateral view)."""
    d = seg[-1] - seg[0]
    ang = np.degrees(np.arctan2(d[1], d[0])) % 180
    if ang < 25 or ang > 155:
        return AP                       # anterior-posterior
    if 65 < ang < 115:
        return SI                     # superior-inferior
    return OB1 if ang < 90 else OB2  # oblique


def chunked(seg, n=14):
    """Split a polyline into overlapping chunks so colour can vary along it."""
    out, i = [], 0
    while i < len(seg) - 2:
        out.append(seg[i:i + n + 1])
        i += n
    return out


def smooth(seg, k=3):
    if len(seg) < 2 * k + 1:
        return seg
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    pad = np.pad(seg, ((k, k), (0, 0)), mode="edge")
    return np.c_[np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")]


# ---------------------------------------- cortex: nested level sets of u
gen = contourpy.contour_generator(x, y, u, name="serial")
for c in [-0.12, 0.18, 0.45]:
    for seg in gen.lines(c):
        ok = sd_at(seg[:, 0], seg[:, 1]) < -0.012
        idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a < 6 or np.ptp(seg[a:b], axis=0).max() < 0.07:
                continue                 # drop specks and tiny islands
            sm = smooth(seg[a:b])
            for ch in chunked(sm):
                op = 0.95 if abs(c) < 0.3 else 0.7
                fibres.append((3, to_d(ch), orient_colour(ch), 1.15 * args.stroke_scale, op))

# outline of the cortex (pial surface), as two close fibres
gen_sd = contourpy.contour_generator(x, y, SD, name="serial")
for c in [-0.004, -0.014]:
    for seg in gen_sd.lines(c):
        for ch in chunked(smooth(seg), 18):
            fibres.append((3, to_d(ch), orient_colour(ch), 1.1 * args.stroke_scale, 0.75))
# silhouette used to hide what lies behind the cerebrum
occluder = "".join(to_d(seg) + "Z" for seg in gen_sd.lines(0.0))


# ---------------------------------------- cerebellum: folia
def g_cerebellum(X, Y):
    u_, v_ = X + 0.62, Y + 0.56
    c, s = np.cos(0.3), np.sin(0.3)
    u_, v_ = c * u_ + s * v_, -s * u_ + c * v_
    return (u_ / 0.42) ** 2 + (v_ / 0.25) ** 2


GC = g_cerebellum(X, Y)
THc = np.arctan2(Y + 0.56, X + 0.62)
uc, vc = X + 0.62, Y + 0.56
cs, sn = np.cos(0.3), np.sin(0.3)
uc, vc = cs * uc + sn * vc, -sn * uc + cs * vc
folia = vc + 0.9 * uc ** 2 + 0.01 * np.sin(18 * uc)
gen_f = contourpy.contour_generator(x, y, folia, name="serial")


if args.cerebellum == "phase":
    # Lateral-view cerebellum, in proportion to the cerebrum (about a third of its
    # length, 30% of its height), its top tucked under the occipital lobe:
    # flat-ish top sloping
    # backwards, full posterior bulge, convex underside with the tonsil, and a
    # hollow front that wraps around the brainstem.
    CBP = [(-0.15, -0.35), (-0.34, -0.31), (-0.56, -0.29), (-0.74, -0.32), (-0.85, -0.40),
           (-0.90, -0.50), (-0.87, -0.61), (-0.79, -0.69), (-0.65, -0.74), (-0.49, -0.76),
           (-0.35, -0.73), (-0.24, -0.68), (-0.17, -0.61), (-0.18, -0.53), (-0.18, -0.44)]
    CB = closed_spline(CBP) - np.array([args.shift_back + args.cerebellum_back, 0.0])
    CBSD = polygon_sdf(X, Y, CB)

    def cb_sd(px, py):
        j = np.clip(np.rint((px - x[0]) / dx).astype(int), 0, nx - 1)
        i = np.clip(np.rint((py - y[0]) / (y[1] - y[0])).astype(int), 0, ny - 1)
        return CBSD[i, j]

    def clip_cb(seg, colour, width, opacity, margin=-0.012):
        ok = cb_sd(seg[:, 0], seg[:, 1]) < margin
        idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a > 3:
                fibres.append((1, to_d(seg[a:b]), colour, width * args.stroke_scale, opacity))

    # FitzHugh-Nagumo, the textbook model of a spiking neuron:
    #   v' = v - v^3/3 - w + I,   w' = eps (v + a - b w)
    # eps = 0.2: still a firing neuron (the rest state is an unstable focus inside
    # a limit cycle), with enough rotation for the flow to read as curves
    a_, b_, eps_, I_ = 0.7, 0.8, 0.2, 0.5

    def fhn(v, w):
        return v - v ** 3 / 3 - w + I_, eps_ * (v + a_ - b_ * w)

    v_, w_ = 2.0, 1.0                                  # find the limit cycle
    h = 0.01
    cyc = []
    for it in range(40000):
        k1 = fhn(v_, w_); k2 = fhn(v_ + h / 2 * k1[0], w_ + h / 2 * k1[1])
        v_, w_ = v_ + h * k2[0], w_ + h * k2[1]
        if it > 32000:
            cyc.append((v_, w_))
    cyc = np.array(cyc)
    # keep exactly one period: the first return close to the starting point
    dist = np.hypot(*(cyc - cyc[0]).T)
    far = np.flatnonzero(dist > 0.5)[0]
    period = far + np.argmin(dist[far:far + int(0.9 * len(cyc))] if len(cyc) > far else dist[far:])
    cyc = cyc[:period + 1]
    lo, hi = cyc.min(0), cyc.max(0)
    # phase plane -> brain: the cycle fills the middle of the cerebellum
    bx0, bx1 = CB[:, 0].min(), CB[:, 0].max()
    by0, by1 = CB[:, 1].min(), CB[:, 1].max()
    ccx, ccy = (bx0 + bx1) / 2 + 0.01, (by0 + by1) / 2 - 0.01
    sx, sy = 0.70 * (bx1 - bx0) / (hi[0] - lo[0]), 0.62 * (by1 - by0) / (hi[1] - lo[1])
    mv, mw = (lo + hi) / 2

    def to_brain(v, w):
        return ccx + sx * (v - mv), ccy + sy * (w - mw)

    def to_phase(px, py):
        return mv + (px - ccx) / sx, mw + (py - ccy) / sy

    def direction(px, py):
        """Unit direction of the FHN flow, drawn in brain coordinates."""
        dv, dw = fhn(*to_phase(px, py))
        bx, by = sx * dv, sy * dw
        n = np.hypot(bx, by) + 1e-12
        return bx / n, by / n

    def streamline(p, length=0.12, step=0.003):
        """Trace the direction field a fixed arc length both ways from p (midpoint RK)."""
        out = []
        for sgn in (-1, 1):
            q, pts = np.array(p, float), []
            for _ in range(int(length / step)):
                d1 = np.array(direction(*q)) * sgn
                d2 = np.array(direction(*(q + 0.5 * step * d1))) * sgn
                q = q + step * d2
                if cb_sd(q[0:1], q[1:2])[0] > -0.006:
                    break
                pts.append(q.copy())
            out.append(pts[::-1] if sgn < 0 else pts)
        return np.array(out[0] + [list(p)] + out[1])

    # even seeding: a jittered grid over the cerebellum, skipping seeds that fall
    # on a line already drawn, so the flow fills the shape with an even texture
    spacing = 0.026
    occupied = np.zeros((ny, nx), bool)
    rad = int(round(0.6 * spacing / dx))

    def mark(seg):
        j = np.clip(np.rint((seg[:, 0] - x[0]) / dx).astype(int), 0, nx - 1)
        i = np.clip(np.rint((seg[:, 1] - y[0]) / (y[1] - y[0])).astype(int), 0, ny - 1)
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                if di * di + dj * dj <= rad * rad:
                    occupied[np.clip(i + di, 0, ny - 1), np.clip(j + dj, 0, nx - 1)] = True

    cyc_b = np.c_[to_brain(cyc[:, 0], cyc[:, 1])]
    mark(cyc_b)
    gxs = np.arange(bx0, bx1, spacing)
    gys = np.arange(by0, by1, spacing)
    seeds = [(gx + rng.uniform(-0.3, 0.3) * spacing, gy + rng.uniform(-0.3, 0.3) * spacing)
             for gy in gys for gx in gxs]
    rng.shuffle(seeds)
    for sxp, syp in seeds:
        if cb_sd(np.array([sxp]), np.array([syp]))[0] > -0.02:
            continue
        j = int(round((sxp - x[0]) / dx)); i = int(round((syp - y[0]) / (y[1] - y[0])))
        if occupied[i, j]:
            continue
        seg = streamline((sxp, syp))
        if len(seg) < 10:
            continue
        mark(seg)
        clip_cb(smooth(seg, 1), orient_colour(seg), 0.75, 0.85)
    # the limit cycle: the spike
    clip_cb(cyc_b, AP, 1.7, 1.0, margin=0.5)
    # outline (cerebellar cortex surface), as two close lines
    gen_cb = contourpy.contour_generator(x, y, CBSD, name="serial")
    for c in (-0.004, -0.014):
        for seg in gen_cb.lines(c):
            for ch in chunked(smooth(seg), 18):
                fibres.append((1, to_d(ch), OB2, 1.0 * args.stroke_scale, 0.8))
for c in (np.linspace(-0.3, 0.45, 30) if args.cerebellum == "folia" else []):
    for seg in gen_f.lines(c + rng.normal(0, 0.003)):
        ok = g_cerebellum(seg[:, 0], seg[:, 1]) < 0.97
        idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a > 4:
                col = AP if rng.random() < 0.55 else (OB1 if rng.random() < 0.5 else OB2)
                fibres.append((1, to_d(smooth(seg[a:b], 2)), col, 0.95 * args.stroke_scale, 0.85))
gen_c = contourpy.contour_generator(x, y, GC, name="serial")
for c in ((0.97, 0.93) if args.cerebellum == "folia" else ()):
    for seg in gen_c.lines(c):
        fibres.append((1, to_d(seg), OB2, 1.0 * args.stroke_scale, 0.8))
if args.no_cerebellum:
    fibres = [f for f in fibres if f[0] != 1]


# ---------------------------------------- brainstem: an integral sign
def catmull_rom(P, n=60):
    P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


ctrl = np.array([[0.30, -0.50], [0.27, -0.43], [0.19, -0.41], [0.11, -0.47],
                 [0.06, -0.58], [-0.03, -0.78], [-0.11, -0.96], [-0.19, -1.07],
                 [-0.28, -1.11], [-0.34, -1.06]])
if args.stem == "anatomical":
    # emerges from under the cerebrum (midbrain), runs down and slightly back,
    # just in front of the cerebellum, and curls back at the medulla
    ctrl = np.array([[0.20, -0.36], [0.17, -0.29], [0.09, -0.27], [0.02, -0.33],
                     [-0.02, -0.46], [-0.06, -0.66], [-0.10, -0.86], [-0.15, -1.00],
                     [-0.23, -1.07], [-0.30, -1.03]]) - np.array([args.shift_back, 0.0])
C = catmull_rom(ctrl)
T = np.gradient(C, axis=0)
T /= np.linalg.norm(T, axis=1)[:, None]
N = np.c_[-T[:, 1], T[:, 0]]
sarc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
sarc /= sarc[-1]
width = 0.012 + 0.07 * np.sin(np.pi * np.clip((sarc - 0.06) / 0.88, 0, 1)) ** 1.3
if args.stem == "anatomical":   # a pons: the stem swells a little below the midbrain
    width = width + 0.018 * np.exp(-((sarc - 0.40) / 0.10) ** 2)
for d in np.linspace(-1, 1, 21):
    seg = C + ((d + rng.normal(0, 0.05)) * width)[:, None] * N
    col = OB1 if abs(d) < 0.35 else (OB2 if abs(d) > 0.8 else SI)
    fibres.append((4, to_d(seg, 2), col, 1.0 * args.stroke_scale, 0.95))
stem_halo = to_d(C, 2)


# ---------------------------------------------------------------- text as paths
def text_d(s, x0, y0, size, prop, tracking=0.0):
    """Outline text into an SVG path, centred on x0, baseline at y0."""
    glyphs, offs, cur = [], [], 0.0
    for ch in s:
        if ch == " ":
            cur += size * 0.35
            continue
        g = TextPath((0, 0), ch, size=size, prop=prop)
        ext = g.get_extents()
        glyphs.append(g)
        offs.append(cur - ext.x0)
        cur += ext.width + tracking * size
    start = x0 - (cur - tracking * size) / 2
    parts = []
    for g, o in zip(glyphs, offs):
        for verts, code in g.iter_segments(curves=True, simplify=False):
            v = [(start + o + verts[i], y0 - verts[i + 1]) for i in range(0, len(verts), 2)]
            cmd = {1: "M", 2: "L", 3: "Q", 4: "C", 79: "Z"}[code]
            parts.append(cmd if code == 79 else cmd + " ".join(f"{a:.2f},{b:.2f}" for a, b in v))
    return "".join(parts)


sans_bold = FontProperties(family="DejaVu Sans", weight="bold")
sans = FontProperties(family="DejaVu Sans")
title = text_d("ICMNS", W / 2, 935, 150, sans_bold, tracking=0.12)
sub = text_d("INTERNATIONAL CONFERENCE ON MATHEMATICAL NEUROSCIENCE",
             W / 2, 995, 22.5, sans, tracking=0.16)

# ---------------------------------------------------------------- write SVG
G = '<g fill="none" stroke-linecap="round" stroke-linejoin="round">'
body = []
for layer in (1, 3, 4):
    if layer == 3:   # cerebrum hides whatever lies behind it
        body.append(f'<path d="{occluder}" fill="{BG}"/>')
    if layer == 4:   # the integral-sign brainstem sits on top, with a dark halo
        body.append(f'<path d="{stem_halo}" fill="none" stroke="{BG}" '
                    f'stroke-width="{2 * 0.085 * S:.0f}" stroke-linecap="round" stroke-opacity="0.85"/>')
    body.append(G)
    body += [f'<path d="{d}" stroke="{c}" stroke-width="{w}" stroke-opacity="{o}"/>'
             for l, d, c, w, o in fibres if l == layer]
    body.append('</g>')

logo = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        '<title>ICMNS: International Conference on Mathematical Neuroscience</title>',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>', *body,
        f'<path d="{title}" fill="{P["TITLE"]}"/>',
        f'<path d="{sub}" fill="{P["SUB"]}"/>', '</svg>']
with open(f"{args.out}_logo.svg", "w") as fh:
    fh.write("\n".join(logo))

vb = (60, 0, 880, 880)                 # brain-only square mark
mark = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{" ".join(map(str, vb))}" '
        f'width="{vb[2]}" height="{vb[3]}">', '<title>ICMNS mark</title>',
        f'<rect x="{vb[0]}" y="{vb[1]}" width="{vb[2]}" height="{vb[3]}" fill="{BG}"/>',
        *body, '</svg>']
with open(f"{args.out}_mark.svg", "w") as fh:
    fh.write("\n".join(mark))
print(len(fibres), "fibre pieces")
