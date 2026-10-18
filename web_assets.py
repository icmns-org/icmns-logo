"""Make web-ready ICMNS logo files from the output of make_logo.py.

  web/icmns_logo.svg      stacked logo (brain above the name), compacted
  web/icmns_mark.svg      brain only, square
  web/icmns_header.svg    horizontal lockup for the site header
  web/favicon.svg         simplified mark with heavier strokes for tiny sizes
  web/*.png, favicon.ico  raster sizes rendered with headless Chromium

The compaction merges every fibre with the same colour/width/opacity into a
single <path>, simplifies each polyline (Ramer-Douglas-Peucker, 0.3 px) and
writes relative coordinates; the picture is unchanged at any normal size.

Run after make_logo.py:   python3 web_assets.py [--src icmns] [--out web] [--transparent]
PNGs are rendered with node + playwright (render.js) if available, otherwise
with a local Google Chrome / Chromium in headless mode (set CHROME to its path).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="icmns", help="prefix used with make_logo.py --out")
ap.add_argument("--out", default="web", help="output folder")
ap.add_argument("--transparent", action="store_true",
                help="no background rectangle (the brain still hides what is behind it, "
                     "using the palette background, so place it on that colour)")
opt = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, opt.out)
os.makedirs(OUT, exist_ok=True)
src = open(os.path.join(HERE, f"{opt.src}_logo.svg")).read()

BG = re.search(r'<rect width="\d+" height="\d+" fill="(#[0-9a-f]+)"', src).group(1)
W, H = map(int, re.search(r'viewBox="0 0 (\d+) (\d+)"', src).groups())


# ---------------------------------------------------------------- parsing
def parse_polylines(d):
    """'M x,y x,y ... [Z] M ...' -> list of (n x 2 arrays, closed?)"""
    out = []
    for sub in re.findall(r"M[^M]+", d):
        closed = sub.rstrip().endswith("Z")
        nums = np.array(re.findall(r"-?\d+\.?\d*", sub), float).reshape(-1, 2)
        out.append((nums, closed))
    return out


def rdp(p, eps):
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.hypot(*ab)
    if n == 0:
        dist = np.hypot(*(p - a).T)
    else:
        dist = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n
    i = int(np.argmax(dist))
    if dist[i] > eps:
        return np.vstack([rdp(p[:i + 1], eps)[:-1], rdp(p[i:], eps)])
    return np.vstack([a, b])


def num(v):
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def encode(polys, eps=0.3):
    """Relative-coordinate path data for a list of polylines."""
    parts = []
    for p, closed in polys:
        p = rdp(p, eps)
        p = np.round(p, 1)
        s = f"M{num(p[0, 0])} {num(p[0, 1])}"
        if len(p) > 1:
            dlt = np.round(np.diff(p, axis=0), 1)
            s += "l" + " ".join(f"{num(a)} {num(b)}" for a, b in dlt)
        parts.append(s + ("z" if closed else ""))
    return "".join(parts).replace(" -", "-")


# ------------------------------------------------- split the generated SVG
groups = re.findall(r'<g fill="none"[^>]*>(.*?)</g>', src, re.S)
layers = []      # per layer: dict style -> list of polylines
for g in groups:
    by_style = defaultdict(list)
    for d, st, sw, so in re.findall(
            r'<path d="([^"]+)" stroke="([^"]+)" stroke-width="([^"]+)" stroke-opacity="([^"]+)"/>', g):
        by_style[(st, sw, so)] += parse_polylines(d)
    layers.append(by_style)
occluder = re.search(rf'<path d="([^"]+)" fill="{BG}"/>', src).group(1)
halo = re.search(r'<path d="([^"]+)" fill="none" stroke="#[0-9a-f]+" stroke-width="(\d+)"', src)
text_paths = re.findall(r'<path d="([^"]+)" fill="(#[0-9a-f]+)"/>\n?(?=<path|</svg>)', src)
title_d, title_fill = text_paths[-2]
sub_d, sub_fill = text_paths[-1]


def brain_body(width_scale=1.0, eps=0.3, min_len=0.0):
    out = []
    for k, by_style in enumerate(layers):
        if k == 1:
            out.append(f'<path d="{encode(parse_polylines(occluder), 0.5)}" fill="{BG}"/>')
        if k == 2:
            out.append(f'<path d="{encode(parse_polylines(halo.group(1)), eps)}" fill="none" '
                       f'stroke="{BG}" stroke-width="{halo.group(2)}" stroke-opacity=".85"/>')
        out.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
        for (st, sw, so), polys in sorted(by_style.items()):
            if min_len:
                polys = [pc for pc in polys if np.ptp(pc[0], axis=0).max() >= min_len]
            if polys:
                out.append(f'<path d="{encode(polys, eps)}" stroke="{st}" '
                           f'stroke-width="{num(float(sw) * width_scale)}" stroke-opacity="{so}"/>')
        out.append("</g>")
    return out


def svg(viewbox, body, title, size=None):
    vb = " ".join(map(str, viewbox))
    wh = f' width="{size[0]}" height="{size[1]}"' if size else ""
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"{wh} role="img">',
                      f"<title>{title}</title>",
                      "" if opt.transparent else
                      f'<rect x="{viewbox[0]}" y="{viewbox[1]}" width="{viewbox[2]}" '
                      f'height="{viewbox[3]}" fill="{BG}"/>', *body, "</svg>"])


def text_bbox(d):
    pts = np.array(re.findall(r"-?\d+\.?\d*", d), float).reshape(-1, 2)
    return pts.min(0), pts.max(0)


NAME = "ICMNS: International Conference on Mathematical Neuroscience"
brain = brain_body()
title = f'<path d="{title_d}" fill="{title_fill}"/>'
sub = f'<path d="{sub_d}" fill="{sub_fill}"/>'

files = {}
files["icmns_logo.svg"] = svg((0, 0, W, H), brain + [title, sub], NAME, (W, H))
MARK_VB = (60, 0, 880, 880)
files["icmns_mark.svg"] = svg(MARK_VB, brain, "ICMNS", (880, 880))

# horizontal lockup: brain on the left, name and subtitle on the right
(tx0, ty0), (tx1, ty1) = text_bbox(title_d)
(sx0, sy0), (sx1, sy1) = text_bbox(sub_d)
pad = 40
bx0, by0, bw, bh = MARK_VB
s_t = 1.6                                   # enlarge the name relative to the brain
tw = (tx1 - tx0) * s_t
s_s = tw / (sx1 - sx0)                      # subtitle spans the name's width
block_h = (ty1 - ty0) * s_t + 34 + (sy1 - sy0) * s_s
x_text = bx0 + bw + 30
y_top = by0 + (bh - block_h) / 2 + 10
lock = brain + [
    f'<g transform="translate({x_text - tx0 * s_t:.1f} {y_top - ty0 * s_t:.1f}) scale({s_t})">{title}</g>',
    f'<g transform="translate({x_text - sx0 * s_s:.1f} '
    f'{y_top + (ty1 - ty0) * s_t + 34 - sy0 * s_s:.1f}) scale({s_s:.4f})">{sub}</g>']
LOCK_VB = (bx0 - pad + 40, by0, int(x_text + tw + pad - (bx0 - pad + 40)), bh)
files["icmns_header.svg"] = svg(LOCK_VB, lock, NAME, LOCK_VB[2:])

# favicon: heavier strokes, fewer tiny pieces, cropped tight to the brain
fav = brain_body(width_scale=6.0, eps=2.0)
files["favicon.svg"] = svg((95, 60, 820, 820), fav, "ICMNS", (820, 820))
# mark for small display sizes (site header, ~48-96 px): moderately heavier strokes
files["icmns_mark_small.svg"] = svg((95, 60, 820, 820), brain_body(width_scale=3.0, eps=1.0),
                                    "ICMNS", (820, 820))

for name, text in files.items():
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)

# ---------------------------------------------------------------- rasters
render = os.path.join(HERE, "render.js")
CHROME = os.environ.get("CHROME") or next(
    (c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 shutil.which("chromium") or "", shutil.which("google-chrome") or ""]
     if c and os.path.exists(c)), None)


def use_node():
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], cwd=HERE,
                          capture_output=True).returncode == 0


NODE = use_node()


def rasterise(svg_path, png_path, w, h):
    """SVG -> PNG at w x h, via playwright if present, else headless Chrome + Pillow."""
    if NODE:
        subprocess.run(["node", render, svg_path, png_path, str(w), str(h)], check=True)
        return
    from PIL import Image
    k = max(1, -(-600 // min(w, h)))            # headless windows can't be tiny: render big
    W2, H2 = w * k, h * k
    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "r.html")
        with open(html, "w") as fh:
            fh.write(f'<html><body style="margin:0;background:transparent">'
                     f'<img src="file://{os.path.abspath(svg_path)}" '
                     f'style="display:block;width:{W2}px;height:{H2}px"></body></html>')
        shot = os.path.join(tmp, "r.png")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--allow-file-access-from-files", "--default-background-color=00000000",
                        # the visible area of a headless window is shorter than the window,
                        # so ask for a taller one and crop: otherwise the bottom stays empty
                        f"--window-size={W2},{H2 + 200}", f"--screenshot={shot}", f"file://{html}"],
                       check=True, capture_output=True)
        Image.open(shot).convert("RGBA").crop((0, 0, W2, H2)).resize(
            (w, h), Image.LANCZOS).save(png_path)

jobs = [  # (svg, png, width, height)
    ("icmns_logo.svg", "icmns_logo_1000.png", 1000, 1080),
    ("icmns_mark.svg", "icmns_mark_512.png", 512, 512),
    ("icmns_header.svg", "icmns_header_1200.png", 1200, None),
    ("favicon.svg", "favicon-32.png", 32, 32),
    ("favicon.svg", "favicon-48.png", 48, 48),
    ("favicon.svg", "apple-touch-icon.png", 180, 180),
    ("favicon.svg", "icon-512.png", 512, 512),
]
for s, p, w, h in jobs:
    if h is None:
        vb = LOCK_VB
        h = round(w * vb[3] / vb[2])
    rasterise(os.path.join(OUT, s), os.path.join(OUT, p), w, h)

# social-share card (Open Graph / Twitter), 1200 x 630
og_w = (LOCK_VB[3] + 120) * 1200 / 630
og_vb = (round(LOCK_VB[0] - (og_w - LOCK_VB[2]) / 2), -60, round(og_w), LOCK_VB[3] + 120)
with open(os.path.join(OUT, "og_card.svg"), "w") as fh:
    fh.write(svg(og_vb, lock, NAME, og_vb[2:]))
rasterise(os.path.join(OUT, "og_card.svg"), os.path.join(OUT, "og-image.png"), 1200, 630)
os.remove(os.path.join(OUT, "og_card.svg"))

from PIL import Image  # noqa: E402  (ships with matplotlib)
Image.open(os.path.join(OUT, "favicon-48.png")).save(
    os.path.join(OUT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)])

for f in sorted(os.listdir(OUT)):
    print(f"{f:28s} {os.path.getsize(os.path.join(OUT, f)) / 1024:8.1f} KB")
