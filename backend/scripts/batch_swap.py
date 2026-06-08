"""Batch hair-swap runner for visual regression.

Runs the in-process pipeline for one base image against every donor image in
a directory, writing the composited PNGs to an output directory plus a
side-by-side contact sheet. Reused for baseline ("old") and improved ("new")
runs so results can be compared image-for-image.

Usage:
    python scripts/batch_swap.py \
        --base ../bald-image.png \
        --donors ../sample_hairstyles \
        --out ../outputs/old
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# Make the backend package importable when run from anywhere.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.model_manager import ModelManager  # noqa: E402
from app.core.settings import get_settings  # noqa: E402
from app.pipeline.builder import build_master_pipeline  # noqa: E402
from app.pipeline.context import ProcessingContext  # noqa: E402

_DONOR_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _to_png(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(rgba, mode="RGBA")


def _contact_row(base: Image.Image, donor: Image.Image, result: Image.Image, height: int = 384) -> Image.Image:
    """Compose [base | donor | result] scaled to a common height."""
    tiles = []
    for img in (base, donor, result):
        rgb = img.convert("RGB")
        w = max(1, int(rgb.width * height / rgb.height))
        tiles.append(rgb.resize((w, height)))
    total_w = sum(t.width for t in tiles) + 2 * 8
    canvas = Image.new("RGB", (total_w, height), (24, 24, 24))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width + 8
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--donors", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warp", default=None, help="override warp impl (tps|mls)")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = out_dir / "_compare"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    models = ModelManager(settings).load()
    pipeline = build_master_pipeline(models, settings)

    overrides: dict[str, str] = {}
    if args.warp:
        overrides["warp"] = args.warp

    base_bytes = _load_bytes(args.base)
    base_img = Image.open(args.base)

    donors = sorted(p for p in args.donors.iterdir() if p.suffix.lower() in _DONOR_EXTS)
    print(f"Base: {args.base.name}  |  {len(donors)} donors  |  out: {out_dir}")
    print(f"Providers: {settings.ort_provider_priority}  warp override: {overrides or 'none'}")

    rows: list[Image.Image] = []
    summary: list[str] = []
    for donor_path in donors:
        name = donor_path.stem
        try:
            ctx = ProcessingContext(
                base_image_bytes=base_bytes,
                donor_image_bytes=_load_bytes(donor_path),
            )
            t0 = time.perf_counter()
            result = pipeline.run(context=ctx, implementation_overrides=overrides or None)
            dt = (time.perf_counter() - t0) * 1000.0
            if result.output_rgba is None:
                raise RuntimeError("no output_rgba")
            out_img = _to_png(result.output_rgba)
            out_img.save(out_dir / f"{name}.png")
            rows.append(_contact_row(base_img, Image.open(donor_path), out_img))
            timings = " ".join(f"{k}={v:.0f}" for k, v in result.timings_ms.items())
            line = f"OK   {name}  total={dt:.0f}ms  [{timings}]"
            print(line)
            summary.append(line)
        except Exception as exc:  # noqa: BLE001 - batch tool, keep going
            line = f"FAIL {name}  {type(exc).__name__}: {exc}"
            print(line)
            summary.append(line)
            traceback.print_exc()

    if rows:
        total_h = sum(r.height for r in rows) + 8 * (len(rows) - 1)
        max_w = max(r.width for r in rows)
        sheet = Image.new("RGB", (max_w, total_h), (12, 12, 12))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 8
        sheet.save(sheet_dir / "contact_sheet.png")
        print(f"\nContact sheet: {sheet_dir / 'contact_sheet.png'}")

    (out_dir / "_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Summary: {out_dir / '_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
