# SciPy 2026 Virtual Poster — Metaxy

LaTeX (`tikzposter`) source for the Metaxy virtual poster. A0 landscape.

Content draws from the Metaxy slide deck (`slides/2026-introducing-metaxy/`)
and the two companion paper drafts in `publications/` (JOSS + SAO workshop).

## Build

```bash
make              # renders any missing assets/*.svg.pdf, then compiles
make distclean    # wipe poster.pdf and rendered SVG PDFs
```

Or manually:

```bash
# one-time: convert SVGs (LaTeX cannot embed SVG natively)
for f in assets/*.svg; do
  inkscape "$f" --export-type=pdf --export-filename="$f.pdf"
done
tectonic poster.tex     # or: latexmk -pdf poster.tex
```

### Why pre-render SVGs?

No LaTeX engine (pdflatex, xelatex, lualatex, tectonic) embeds SVG directly.
The `svg` package papers over this by shelling out to `inkscape` at compile
time, but that path needs `\write18` piping — which tectonic does not
implement. Pre-rendering with inkscape is portable across engines and keeps
builds deterministic; the `Makefile` treats each `.svg.pdf` as a target so
rendering only re-runs when a source SVG changes.

Verified on macOS with `tectonic 0.15.0` and `inkscape 1.4`.

## Layout

- **Hero band** (dark navy) — title, tagline, authors, install command.
- **3-column body** (cream / white blocks):
  1. GPU economics problem + state-of-the-field comparison.
  2. Key insight, hierarchical versioning, field-level dependency diagram.
  3. Code (define · ask · record), pluggable backends, correctness tests.
- **Dark band** — production story at Anam + three-takeaway fade.
- **Footer** — authors, references, live-session hooks.

## Assets

`assets/` holds **symlinks** into `docs/assets/publications/2026-introducing-metaxy/`
and `docs/assets/metaxy.svg`; nothing is copied. Edit upstream, re-render
with `make assets`.

## Dependencies

- `tectonic` (recommended) or a full TeX Live with `tikzposter`, `ebgaramond`,
  `sourcesanspro`, `sourcecodepro`, `fancyvrb`, `url`.
- `inkscape` for SVG → PDF conversion.

### Package choices

- **No `hyperref`** — `tikzposter` + `hyperref` raises *"Loading a class or
  package in a group"* at `\begin{document}` under TeX Live 2026 pdflatex.
  We use plain `url` for typesetting URLs and a stubbed `\href{url}{text}` →
  `text` so the source still uses the familiar macro shape. Explicit URLs
  that need to be visible on print are wrapped in `\url{...}` in the
  References block.
- **No `amssymb`** — only `amsmath` is needed for `\to`. Skipping `amssymb`
  keeps `umsa.fd` / `umsb.fd` out of the begin-document hook chain.
- **No `[default]` option to `sourcesanspro`** — that option pulls in
  `fontspec` under pdflatex, which conflicts with `tikzposter`'s own setup.
  We set `\familydefault` to `\sfdefault` manually instead.
