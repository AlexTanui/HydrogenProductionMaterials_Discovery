"""Export a notebook's figures to PNG files, without re-running it.

An executed `.ipynb` already stores every figure as base64 PNG inside its
output cells, so the images can be lifted straight out of the file. That
matters here: re-executing `notebooks/eda_md17.ipynb` needs all nine
gold-stage files (several hundred MB, regenerated per machine), and the
images would come out slightly different anyway - the notebook samples
configurations with a generator that advances as cells run, so a re-run
is not guaranteed to reproduce the committed figures pixel for pixel.
Extracting gives exactly the figures the write-up in `docs/eda.md`
describes.

    python -m scripts.export_notebook_figures                       # eda_md17
    python -m scripts.export_notebook_figures --notebook notebooks/eda_md22.ipynb

Each figure is named from the markdown heading above it, so
`## 2. Energy distributions` becomes `md17_02_energy_distributions.png`
rather than `output_5_0.png`. Re-running overwrites in place, so the
files stay in step with the notebook.

`reports/figures/` is tracked by git (only `reports/*.pdf` is ignored),
so the exported PNGs can be committed and pulled by anyone building a
deck - no notebook, no data, no Python needed on their side.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "eda_md17.ipynb"
DEFAULT_OUT = ROOT / "reports" / "figures"


def slugify(text: str, max_words: int = 6) -> str:
    """`## 4. Interatomic distances vs. the 5 Å cutoff radius`
    -> `interatomic_distances_vs_the_5_cutoff`"""
    text = re.sub(r"^#+\s*", "", text.strip())
    text = re.sub(r"^\d+[.)]\s*", "", text)          # drop the section number
    text = text.split("\n")[0]
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)      # punctuation and non-ASCII to spaces
    words = [w.lower() for w in text.split() if w]
    return "_".join(words[:max_words]) or "figure"


def heading_number(text: str) -> str:
    """The `2` in `## 2. Energy distributions`, as a zero-padded prefix, so
    the files sort in the notebook's own reading order."""
    m = re.match(r"^#+\s*(\d+)", text.strip())
    return f"{int(m.group(1)):02d}" if m else "00"


def export(notebook_path: Path, out_dir: Path, prefix: str | None = None,
           dry_run: bool = False) -> list[Path]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    prefix = prefix or notebook_path.stem.replace("eda_", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    section_slug, section_num = "figure", "00"
    seen: dict[str, int] = {}
    written: list[Path] = []

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))

        if cell.get("cell_type") == "markdown":
            # Remember the most recent heading; it names the figures below it.
            for line in source.splitlines():
                if line.lstrip().startswith("#"):
                    section_slug = slugify(line)
                    section_num = heading_number(line)
                    break
            continue

        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if not png:
                continue

            stem = f"{prefix}_{section_num}_{section_slug}"
            n = seen.get(stem, 0)
            seen[stem] = n + 1
            # A section with two figures gets _b, _c - the first keeps the
            # plain name so existing references to it do not break.
            name = stem if n == 0 else f"{stem}_{chr(ord('b') + n - 1)}"
            path = out_dir / f"{name}.png"

            if not dry_run:
                data = png if isinstance(png, str) else "".join(png)
                path.write_bytes(base64.b64decode(data))
            written.append(path)

    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--prefix", default=None,
                   help="filename prefix (default: notebook stem without 'eda_')")
    p.add_argument("--dry-run", action="store_true", help="list what would be written")
    args = p.parse_args(argv)

    notebook = Path(args.notebook)
    if not notebook.exists():
        raise SystemExit(f"no notebook at {notebook}")

    written = export(notebook, Path(args.out_dir), args.prefix, args.dry_run)

    if not written:
        raise SystemExit(
            f"{notebook.name} has no embedded images. It was probably committed "
            "without outputs - execute it first:\n"
            f"  jupyter nbconvert --to notebook --execute --inplace {notebook}"
        )

    verb = "would write" if args.dry_run else "wrote"
    for path in written:
        size = "" if args.dry_run else f"  ({path.stat().st_size // 1024} KB)"
        print(f"{verb} {path.relative_to(ROOT)}{size}")
    print(f"\n{len(written)} figure(s) from {notebook.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
