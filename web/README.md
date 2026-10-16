# ICMNS logo: web files

| File | Use |
|---|---|
| `icmns_header.svg` (81 KB) | Horizontal logo for the site header |
| `icmns_logo.svg` (81 KB) | Stacked logo (brain above the name), e.g. the home page |
| `icmns_mark.svg` (67 KB) | Brain only, square |
| `favicon.svg`, `favicon.ico`, `favicon-32.png`, `favicon-48.png` | Browser tab icons |
| `apple-touch-icon.png` (180 px), `icon-512.png` | Phone home screens, web app manifest |
| `og-image.png` (1200 × 630) | Preview card when the site is shared on social media or chat apps |
| `*_1000.png`, `*_512.png`, `*_1200.png` | Raster copies for slides, emails and documents |

All SVGs carry their own navy background (`#0b1021`); a page or header in the same colour blends seamlessly.

## HTML `<head>` snippet

```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<meta property="og:image" content="https://www.icmns.org/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#0b1021">
```

Header image:

```html
<a href="/"><img src="/icmns_header.svg" alt="ICMNS: International Conference on Mathematical Neuroscience" height="80"></a>
```

## Regenerating

```sh
python3 make_logo.py                 # add --mask brain.png, --palette warm, --seed N as wanted
python3 web_assets.py                # rebuilds everything in web/
```

`web_assets.py` renders the PNGs with headless Chromium via Node's `playwright` package (`npm i playwright && npx playwright install chromium`).
