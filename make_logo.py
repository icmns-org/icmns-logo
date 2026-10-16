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
}
args = argparse.ArgumentParser(description=__doc__.split("\n")[0])
args.add_argument("--palette", default="cool", choices=PALETTES)
args.add_argument("--mask", help="image whose shape gives the cortex outline "
                  "(PNG with transparency, or a dark/light silhouette)")
args.add_argument("--seed", type=int, default=3)
args.add_argument("--no-cerebellum", action="store_true",
                  help="omit the cerebellum (e.g. if the mask image already includes it)")
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


SD = image_sdf(args.mask) if args.mask else cortex_sdf(X, Y)


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
                fibres.append((3, to_d(ch), orient_colour(ch), 1.15, op))

# outline of the cortex (pial surface), as two close fibres
gen_sd = contourpy.contour_generator(x, y, SD, name="serial")
for c in [-0.004, -0.014]:
    for seg in gen_sd.lines(c):
        for ch in chunked(smooth(seg), 18):
            fibres.append((3, to_d(ch), orient_colour(ch), 1.1, 0.75))
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
for c in np.linspace(-0.3, 0.45, 30):
    for seg in gen_f.lines(c + rng.normal(0, 0.003)):
        ok = g_cerebellum(seg[:, 0], seg[:, 1]) < 0.97
        idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a > 4:
                col = AP if rng.random() < 0.55 else (OB1 if rng.random() < 0.5 else OB2)
                fibres.append((1, to_d(smooth(seg[a:b], 2)), col, 0.95, 0.85))
gen_c = contourpy.contour_generator(x, y, GC, name="serial")
for c in (0.97, 0.93):
    for seg in gen_c.lines(c):
        fibres.append((1, to_d(seg), OB2, 1.0, 0.8))
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
C = catmull_rom(ctrl)
T = np.gradient(C, axis=0)
T /= np.linalg.norm(T, axis=1)[:, None]
N = np.c_[-T[:, 1], T[:, 0]]
sarc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
sarc /= sarc[-1]
width = 0.012 + 0.07 * np.sin(np.pi * np.clip((sarc - 0.06) / 0.88, 0, 1)) ** 1.3
for d in np.linspace(-1, 1, 21):
    seg = C + ((d + rng.normal(0, 0.05)) * width)[:, None] * N
    col = OB1 if abs(d) < 0.35 else (OB2 if abs(d) > 0.8 else SI)
    fibres.append((4, to_d(seg, 2), col, 1.0, 0.95))
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
        f'<path d="{title}" fill="{P['TITLE']}"/>',
        f'<path d="{sub}" fill="{P['SUB']}"/>', '</svg>']
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
