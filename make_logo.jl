#=
ICMNS logo generator (Julia port of make_logo.py).

Lateral view of a convoluted brain, drawn as fibres:
  * gyri and sulci are a Turing pattern: a solution of the Swift-Hohenberg
    equation  u_t = r u - (1 + Lap/k0^2)^2 u - u^3  on the cortical domain;
  * each gyrus is drawn as nested level sets {u = c}, coloured by fibre
    direction as in diffusion-tensor tractography;
  * cerebellar folia are level sets of a curved coordinate;
  * the brainstem is a bundle of parallel offsets of an integral sign.

Usage:
    julia make_logo.jl [--palette cool|warm] [--mask brain.png] [--seed 3]
                       [--no-cerebellum] [--out icmns]

Packages (install once):
    julia -e 'using Pkg; Pkg.add(["FFTW", "Contour", "FileIO", "ImageIO", "ColorTypes"])'
FileIO/ImageIO/ColorTypes are only loaded when --mask is given.
Random numbers differ from NumPy's, so the folding differs from the Python
version for the same seed; try a few seeds and keep the one you like.
=#
using FFTW, Contour, Random, Printf

include(joinpath(@__DIR__, "icmns_text_paths.jl"))   # TITLE_D, SUB_D

# ------------------------------------------------------------ options
function parse_args(argv)
    o = Dict{String,Any}("palette" => "cool", "mask" => nothing, "seed" => 3,
                         "no-cerebellum" => false, "out" => "icmns")
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--no-cerebellum"
            o["no-cerebellum"] = true
        elseif a in ("--palette", "--mask", "--seed", "--out")
            o[a[3:end]] = a == "--seed" ? parse(Int, argv[i+1]) : argv[i+1]
            i += 1
        else
            error("unknown option $a")
        end
        i += 1
    end
    o
end
const OPT = parse_args(ARGS)
if OPT["mask"] !== nothing
    using FileIO, ColorTypes
end

# colour roles: AP = anterior-posterior fibres, SI = superior-inferior,
# OB1/OB2 = the two oblique directions (as in DTI colour coding)
const PALETTES = Dict(
    "warm" => (BG="#1e1c1d", AP="#ef6b5f", SI="#9fe07c", OB1="#e7ef7a", OB2="#3f9bd6",
               TITLE="#f2f0ec", SUB="#bdb8b0"),
    "cool" => (BG="#0b1021", AP="#4c8dff", SI="#3fe0c5", OB1="#8fd8ff", OB2="#9b7bff",
               TITLE="#eef4ff", SUB="#9fb0cc"),
)
const P = PALETTES[OPT["palette"]]
const rng = MersenneTwister(OPT["seed"])

const W, H = 1000, 1080              # SVG canvas
const CX, CY, S = 500.0, 400.0, 345.0   # brain origin and scale (px per unit)

# ------------------------------------------------------------ domain
# arrays are indexed [i, j] = (x[i], y[j]), the layout Contour.jl expects
const nx, ny = 640, 480
const xs = collect(range(-1.35, 1.35; length=nx))
const dx = xs[2] - xs[1]
const ys = collect(range(-0.95, 1.07; length=ny))
const dy = ys[2] - ys[1]
const X = [x for x in xs, _ in ys]
const Y = [y for _ in xs, y in ys]

smin(a, b, k=30.0) = -log(exp(-k * a) + exp(-k * b)) / k
smax(a, b, k=30.0) = -smin(-a, -b, k)

function seg_dist(x, y, p, q)
    d = (q[1] - p[1], q[2] - p[2])
    t = clamp(((x - p[1]) * d[1] + (y - p[2]) * d[2]) / (d[1]^2 + d[2]^2), 0, 1)
    hypot(x - p[1] - t * d[1], y - p[2] - t * d[2]), t
end

"Approximate signed 'distance' (< 0 inside) of the cerebral cortex."
function cortex_sdf(x, y)
    e = sqrt((x / 1.16)^2 + ((y - 0.12) / 0.80)^2) - 1.0      # hemisphere
    floor_ = (-0.22 + 0.13 * x) - y                              # flat underside
    upper = smax(0.8 * e, floor_)
    c, s = cos(-0.18), sin(-0.18)                                # temporal lobe
    u, v = x - 0.02, y + 0.36
    u, v = c * u + s * v, -s * u + c * v
    temporal = sqrt((u / 0.74)^2 + (v / 0.27)^2) - 1.0
    brain = smin(upper, 0.3 * temporal, 40.0)
    d, t = seg_dist(x, y, (0.75, -0.2), (-0.12, -0.02))          # lateral fissure
    smax(brain, -(d - (0.012 + 0.018 * t)), 60.0)
end

kfreq(n, h) = 2π .* fftfreq(n, 1 / h)

"Cortex outline from an image: scaled to the built-in outline's box; a blurred
indicator stands in for a signed distance near the edge."
function image_sdf(path)
    img = FileIO.load(path)
    lum(c) = c isa ColorTypes.AbstractGray ? Float64(ColorTypes.gray(c)) :
        (Float64(ColorTypes.red(c)) + Float64(ColorTypes.green(c)) + Float64(ColorTypes.blue(c))) / 3
    alp(c) = Float64(ColorTypes.alpha(c))
    A = alp.(img)
    shape = if minimum(A) < 0.5
        A .> 0.5                                         # transparency
    else
        L = lum.(img)
        border = vcat(L[1, :], L[end, :], L[:, 1], L[:, end])
        bgl = sort(border)[cld(length(border), 2)]
        abs.(L .- bgl) .> 0.25                           # differs from the border
    end
    rows = findall(vec(any(shape; dims=2)))
    cols = findall(vec(any(shape; dims=1)))
    shape = shape[first(rows):last(rows), first(cols):last(cols)]
    h, w = size(shape)                                   # row 1 = top
    half_w = 1.18
    half_h = half_w * h / w
    y0 = 0.12 - (half_h - 0.80)                          # keep the top at y = 0.92
    ind = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        r = h - floor(Int, (ys[j] - (y0 - half_h)) / (2half_h) * h)   # flip: y up
        c = floor(Int, (xs[i] + half_w) / (2half_w) * w) + 1
        (1 <= r <= h && 1 <= c <= w) && (ind[i, j] = shape[r, c])
    end
    σ = 0.02
    kx, ky = kfreq(nx, dx), kfreq(ny, dy)
    G = [exp(-0.5σ^2 * (a^2 + b^2)) for a in kx, b in ky]
    blur = real(ifft(fft(ind) .* G))
    (0.5 .- blur) .* (2σ * 1.25)
end

const SD = OPT["mask"] === nothing ? cortex_sdf.(X, Y) : image_sdf(OPT["mask"])

"SD at an arbitrary point (nearest grid node)."
sd_at(x, y) = SD[clamp(round(Int, (x - xs[1]) / dx) + 1, 1, nx),
                 clamp(round(Int, (y - ys[1]) / dy) + 1, 1, ny)]

# ------------------------------------------- Swift-Hohenberg on the cortex
function swift_hohenberg()
    k0 = 2π / 0.19                         # gyral wavelength (brain units)
    kx, ky = kfreq(nx, dx), kfreq(ny, dy)
    Lop = [0.3 - (1 - (a^2 + b^2) / k0^2)^2 for a in kx, b in ky]
    dt = 0.5
    denom = 1 .- dt .* Lop
    u = 0.3 .* randn(rng, nx, ny)
    pin = clamp.(-SD ./ 0.02, 0, 1)
    F = plan_fft(u); Fi = plan_ifft(complex(u))
    n_it = 170
    for it in 0:n_it-1
        û = (F * u .+ dt .* (F * (-u .^ 3))) ./ denom
        u = real(Fi * û)
        if it > n_it - 60                 # pin only at the end: keep the labyrinth
            u = pin .* u .+ (1 .- pin) .* (-0.5)
        end
    end
    u
end
const U = swift_hohenberg()

# ------------------------------------------------------------ helpers
const fibres = Tuple{Int,String,String,Float64,Float64}[]   # layer, d, colour, width, opacity

pts(line) = (c = coordinates(line); hcat(c[1], c[2]))      # n x 2 matrix

function to_d(seg; step=1)
    n = size(seg, 1)
    idx = n > 3step ? (1:step:n) : (1:n)
    io = IOBuffer()
    print(io, "M")
    for (k, i) in enumerate(idx)
        k > 1 && print(io, " ")
        @printf(io, "%.1f,%.1f", CX + S * seg[i, 1], CY - S * seg[i, 2])
    end
    String(take!(io))
end

"DTI-style colour from the mean fibre direction (lateral view)."
function orient_colour(seg)
    d = seg[end, :] .- seg[1, :]
    ang = mod(rad2deg(atan(d[2], d[1])), 180)
    (ang < 25 || ang > 155) && return P.AP       # anterior-posterior
    65 < ang < 115 && return P.SI                 # superior-inferior
    ang < 90 ? P.OB1 : P.OB2                      # oblique
end

"Split a polyline into overlapping chunks so colour can vary along it."
function chunked(seg, n=14)
    out, i, m = Matrix{Float64}[], 1, size(seg, 1)
    while i < m - 1
        push!(out, seg[i:min(i + n, m), :])
        i += n
    end
    out
end

function smooth(seg, k=3)
    m = size(seg, 1)
    m < 2k + 1 && return seg
    pad = vcat(repeat(seg[1:1, :], k), seg, repeat(seg[end:end, :], k))
    [sum(pad[i:i+2k, c]) / (2k + 1) for i in 1:m, c in 1:2]
end

"Index ranges (runs) where the boolean vector is true."
function runs(ok)
    r, s = UnitRange{Int}[], 0
    for (i, v) in enumerate(ok)
        v && s == 0 && (s = i)
        if !v && s > 0
            push!(r, s:i-1); s = 0
        end
    end
    s > 0 && push!(r, s:length(ok))
    r
end

# ------------------------------------------- cortex: nested level sets of u
for c in (-0.12, 0.18, 0.45)
    for line in lines(contour(xs, ys, U, c))
        seg = pts(line)
        ok = [sd_at(seg[i, 1], seg[i, 2]) < -0.012 for i in 1:size(seg, 1)]
        for r in runs(ok)
            part = seg[r, :]
            ext = maximum(maximum(part; dims=1) .- minimum(part; dims=1))
            (length(r) < 6 || ext < 0.07) && continue   # drop specks and tiny islands
            for ch in chunked(smooth(part))
                push!(fibres, (3, to_d(ch), orient_colour(ch), 1.15, abs(c) < 0.3 ? 0.95 : 0.7))
            end
        end
    end
end

# outline of the cortex (pial surface), as two close fibres
for c in (-0.004, -0.014), line in lines(contour(xs, ys, SD, c))
    for ch in chunked(smooth(pts(line)), 18)
        push!(fibres, (3, to_d(ch), orient_colour(ch), 1.1, 0.75))
    end
end
# silhouette used to hide what lies behind the cerebrum
const occluder = join(to_d(pts(l)) * "Z" for l in lines(contour(xs, ys, SD, 0.0)))

# ------------------------------------------- cerebellum: folia
function cbl_coords(x, y)
    u, v = x + 0.62, y + 0.56
    c, s = cos(0.3), sin(0.3)
    c * u + s * v, -s * u + c * v
end
g_cerebellum(x, y) = ((u, v) = cbl_coords(x, y); (u / 0.42)^2 + (v / 0.25)^2)

if !OPT["no-cerebellum"]
    folia = [(uv = cbl_coords(x, y); uv[2] + 0.9uv[1]^2 + 0.01sin(18uv[1])) for x in xs, y in ys]
    for c in range(-0.3, 0.45; length=30)
        for line in lines(contour(xs, ys, folia, c + 0.003randn(rng)))
            seg = pts(line)
            ok = [g_cerebellum(seg[i, 1], seg[i, 2]) < 0.97 for i in 1:size(seg, 1)]
            for r in runs(ok)
                length(r) > 4 || continue
                col = rand(rng) < 0.55 ? P.AP : (rand(rng) < 0.5 ? P.OB1 : P.OB2)
                push!(fibres, (1, to_d(smooth(seg[r, :], 2)), col, 0.95, 0.85))
            end
        end
    end
    GCB = g_cerebellum.(X, Y)
    for c in (0.97, 0.93), line in lines(contour(xs, ys, GCB, c))
        push!(fibres, (1, to_d(pts(line)), P.OB2, 1.0, 0.8))
    end
end

# ------------------------------------------- brainstem: an integral sign
function catmull_rom(Pm; n=60)
    Pp = vcat(Pm[1:1, :], Pm, Pm[end:end, :])
    out = Matrix{Float64}[]
    t = collect(range(0, 1; length=n + 1))[1:end-1]
    for i in 2:size(Pp, 1)-2
        p0, p1, p2, p3 = (Pp[i+k, :]' for k in -1:2)
        push!(out, 0.5 .* ((2p1) .+ (-p0 .+ p2) .* t .+ (2p0 .- 5p1 .+ 4p2 .- p3) .* t .^ 2
                           .+ (-p0 .+ 3p1 .- 3p2 .+ p3) .* t .^ 3))
    end
    push!(out, Pp[end-1:end-1, :])
    vcat(out...)
end

ctrl = [0.30 -0.50; 0.27 -0.43; 0.19 -0.41; 0.11 -0.47; 0.06 -0.58;
        -0.03 -0.78; -0.11 -0.96; -0.19 -1.07; -0.28 -1.11; -0.34 -1.06]
C = catmull_rom(ctrl)
m = size(C, 1)
T = similar(C)
for i in 1:m                              # central differences, like np.gradient
    a, b = max(i - 1, 1), min(i + 1, m)
    T[i, :] = (C[b, :] .- C[a, :]) ./ (b - a)
end
T ./= sqrt.(sum(T .^ 2; dims=2))
N = hcat(-T[:, 2], T[:, 1])
sarc = vcat(0.0, cumsum(sqrt.(sum(diff(C; dims=1) .^ 2; dims=2))[:]))
sarc ./= sarc[end]
width = 0.012 .+ 0.07 .* sin.(π .* clamp.((sarc .- 0.06) ./ 0.88, 0, 1)) .^ 1.3
for d in range(-1, 1; length=21)
    seg = C .+ ((d + 0.05randn(rng)) .* width) .* N
    col = abs(d) < 0.35 ? P.OB1 : (abs(d) > 0.8 ? P.OB2 : P.SI)
    push!(fibres, (4, to_d(seg; step=2), col, 1.0, 0.95))
end
stem_halo = to_d(C; step=2)

# ------------------------------------------------------------ write SVG
shuffle!(rng, fibres)
body = String[]
for layer in (1, 3, 4)
    layer == 3 && push!(body, """<path d="$occluder" fill="$(P.BG)"/>""")
    layer == 4 && push!(body, """<path d="$stem_halo" fill="none" stroke="$(P.BG)" """ *
        """stroke-width="$(round(Int, 2 * 0.085 * S))" stroke-linecap="round" stroke-opacity="0.85"/>""")
    push!(body, """<g fill="none" stroke-linecap="round" stroke-linejoin="round">""")
    for (l, d, c, w, o) in fibres
        l == layer && push!(body, """<path d="$d" stroke="$c" stroke-width="$w" stroke-opacity="$o"/>""")
    end
    push!(body, "</g>")
end

logo = vcat(["""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $W $H" width="$W" height="$H">""",
             "<title>ICMNS: International Conference on Mathematical Neuroscience</title>",
             """<rect width="$W" height="$H" fill="$(P.BG)"/>"""], body,
            ["""<path d="$TITLE_D" fill="$(P.TITLE)"/>""",
             """<path d="$SUB_D" fill="$(P.SUB)"/>""", "</svg>"])
write("$(OPT["out"])_logo.svg", join(logo, "\n"))

vb = (60, 0, 880, 880)                    # brain-only square mark
mark = vcat(["""<svg xmlns="http://www.w3.org/2000/svg" viewBox="$(join(vb, " "))" width="$(vb[3])" height="$(vb[4])">""",
             "<title>ICMNS mark</title>",
             """<rect x="$(vb[1])" y="$(vb[2])" width="$(vb[3])" height="$(vb[4])" fill="$(P.BG)"/>"""],
            body, ["</svg>"])
write("$(OPT["out"])_mark.svg", join(mark, "\n"))
println(length(fibres), " fibre pieces")
