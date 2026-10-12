#!/usr/bin/env python3
"""
annotate_ssa.py - automatic pixel-level annotation of SSA (space situational
awareness) FITS imagery: stars/blobs and streaks, exported as YOLO-Ultralytics
segmentation labels on 1024x1024 tiles.

Pipeline (per FITS image)
  1. read FITS (16-bit, single band)
  2. pre-process: mesh background + noise map, hot-pixel repair, background
     flattening, global linear stretch to 8-bit
  3. detect on the WHOLE image (so objects are never cut by tile borders):
     Gaussian matched filter -> SNR map -> hysteresis threshold -> connected
     components -> per-object refinement -> geometry measurement -> class
  4. tile to 1024x1024 with edge-aligned last row/column (no padding, no data loss)
  5. write YOLO segmentation polygons per tile (class 0 = blob/star, 1 = streak)
  6. re-patch tiles to the full-size image, verify it is bit-identical to the
     processed image, and draw the labels read back FROM THE LABEL FILES on top
     of it (so the overlay really shows what is exported)

Usage
  python annotate_ssa.py --input ./Datasets_Assessment --output ./output
  python annotate_ssa.py --input img.fits --output ./tune --roi 2000 1500 1024 1024   # quick tuning on a crop
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

try:
    from astropy.io import fits
except ImportError: 
    fits = None

CLASS_NAMES = {0: "blob", 1: "streak"}  # blob == star
# BGR overlay colours (blob = purple/blue, streak = red, like the reference figure)
COLOURS = {0: (255, 90, 90), 1: (60, 60, 255)}


# --------------------------------------------------------------------------- IO
def _read_fits_minimal(path):
    """Tiny primary-HDU reader (2-D, BITPIX 8/16/32/-32/-64, BZERO/BSCALE)."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            block = f.read(2880)
            done = False
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "ignore")
                key = card[:8].strip()
                if key == "END":
                    done = True
                    break
                if card[8:10] == "= ":
                    hdr[key] = card[10:].split("/")[0].strip()
            if done:
                break
        bitpix, nx, ny = int(hdr["BITPIX"]), int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
        dt = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}[bitpix]
        data = np.frombuffer(f.read(nx * ny * abs(bitpix) // 8), dtype=dt).reshape(ny, nx)
    bzero, bscale = float(hdr.get("BZERO", 0)), float(hdr.get("BSCALE", 1))
    if bitpix == 16 and bzero == 32768 and bscale == 1:
        data = (data.astype(np.int32) + 32768).astype(np.uint16)
    elif bzero != 0 or bscale != 1:
        data = data.astype(np.float32) * bscale + bzero
    return data, hdr


def read_fits(path):
    """Return the first 2-D image found in the FITS file as a numpy array."""
    if fits is None:
        print("  (astropy not installed - using minimal FITS reader)")
        return _read_fits_minimal(path)
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = np.asarray(hdu.data)
                while data.ndim > 2:
                    data = data[0]
                return data, hdu.header
    raise ValueError(f"No image data in {path}")


def mesh_stats(img, box):
    """Robust (sigma-clipped) background level and noise sigma on a coarse mesh."""
    H, W = img.shape
    ny, nx = math.ceil(H / box), math.ceil(W / box)
    bg = np.empty((ny, nx), np.float32)
    sg = np.empty((ny, nx), np.float32)
    for j in range(ny):
        for i in range(nx):
            blk = img[j * box:(j + 1) * box, i * box:(i + 1) * box][::2, ::2].ravel()
            blk = blk.astype(np.float32)
            med = float(np.median(blk))
            sig = 1.4826 * float(np.median(np.abs(blk - med)))
            if sig <= 0:
                sig = float(blk.std())
            for _ in range(3):  # iterative 3-sigma clipping
                keep = blk[np.abs(blk - med) <= 3 * max(sig, 1e-6)]
                if keep.size < 10:
                    break
                med, sig = float(np.median(keep)), float(keep.std())
            bg[j, i], sg[j, i] = med, max(sig, 1e-3)
    if ny >= 3 and nx >= 3:  # suppress outlier meshes (big stars/streaks)
        bg = ndi.median_filter(bg, size=3, mode="nearest")
        sg = ndi.median_filter(sg, size=3, mode="nearest")
    return bg, sg


def upsample(mesh, H, W):
    if mesh.shape == (1, 1):
        return np.full((H, W), mesh[0, 0], np.float32)
    return cv2.resize(mesh, (W, H), interpolation=cv2.INTER_LINEAR)


def preprocess(raw, a):
    """Return (flat8, Z, info). flat8: processed 8-bit image, Z: detection SNR map."""
    img = raw.astype(np.float32)
    H, W = img.shape
    bg_m, sg_m = mesh_stats(img, a.mesh)
    bg, sg = upsample(bg_m, H, W), upsample(sg_m, H, W)

    # hot / isolated single pixels: far above the background while all 8
    # neighbours are at noise level -> replace by the 3x3 median
    n_hot = 0
    if a.hot_sigma > 0:
        k = np.ones((3, 3), np.uint8)
        k[1, 1] = 0
        nbmax = cv2.dilate(img, k)
        hot = (img > bg + a.hot_sigma * sg) & (nbmax < bg + 2.5 * sg)
        n_hot = int(hot.sum())
        if n_hot:
            med = cv2.medianBlur(img, 3)
            img[hot] = med[hot]
            del med
        del nbmax, hot

    # background flattening (keeps the global sky level so the image looks natural)
    bg_ref = float(np.median(bg_m))
    sig_ref = float(np.median(sg_m))
    img -= bg
    img += bg_ref
    del bg, sg

    # global linear stretch -> 8 bit (same scaling for every tile of the image)
    lo = bg_ref - a.stretch[0] * sig_ref
    hi = bg_ref + a.stretch[1] * sig_ref
    flat8 = np.clip((img - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # matched filter + SNR map for detection
    S = cv2.GaussianBlur(img, (0, 0), a.smooth)
    del img
    bg2_m, sg2_m = mesh_stats(S, a.mesh)
    S -= upsample(bg2_m, H, W)
    S /= upsample(sg2_m, H, W)
    info = dict(bg=bg_ref, sigma=sig_ref, hot_pixels=n_hot,
                stretch_lo=lo, stretch_hi=hi)
    return flat8, S, info


# ------------------------------------------------------------------ detection
def measure(m):
    """Geometry of a boolean mask crop: length L, width W (px) from min-area rectangle."""
    ys, xs = np.nonzero(m)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    L, Wd = max(w, h) + 1.0, min(w, h) + 1.0  # +1: pixel extent, not centre distance
    return dict(area=int(xs.size), L=L, W=Wd, elong=L / Wd, angle=float(ang),
                cx=float(xs.mean()), cy=float(ys.mean()))


def classify(g, a):
    """Class 1 = streak, 0 = blob(star) + a review flag for borderline objects."""
    L, e = g["L"], g["elong"]
    if L >= a.streak_len and e >= a.streak_elong:
        return 1, ("short_streak_review" if L < 1.3 * a.streak_len else "")
    if L >= a.fat_len and e >= a.fat_elong:
        return 1, "fat_streak"
    if e >= 1.6 and L >= 8:
        return 0, "elongated_blob_review"
    return 0, ""


def detect(Z, a):
    """Hysteresis detection on the SNR map. Returns (inst, cls_of, catalogue)."""
    H, W = Z.shape
    low = (Z > a.low).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    del low
    inst = np.zeros((H, W), np.int32)
    cls_of, cat = [None], []
    if n <= 1:
        return inst, cls_of, cat
    peaks = ndi.maximum(Z, labels, np.arange(1, n))
    areas = stats[1:, cv2.CC_STAT_AREA]
    sel = np.nonzero((peaks >= a.seed) & (areas >= a.min_area) & (areas <= a.max_area))[0] + 1
    for lab in sel:
        x, y, w, h = (int(v) for v in stats[lab, :4])
        sl = (slice(y, y + h), slice(x, x + w))
        zc = Z[sl]
        m = labels[sl] == lab
        peak = float(zc[m].max())
        # per-object refinement: bright objects get a tighter outline (peak_frac * peak)
        m &= zc >= max(a.low, a.peak_frac * peak)
        nn, lb, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if nn > 2:  # keep the dominant piece only
            m = lb == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        if m.sum() < a.min_area:
            continue
        g = measure(m)
        klass, flag = classify(g, a)
        oid = len(cls_of)
        cls_of.append(klass)
        inst[sl][m] = oid
        edge = x == 0 or y == 0 or x + w == W or y + h == H
        cat.append(dict(id=oid, cls=CLASS_NAMES[klass], cx=round(x + g["cx"], 1),
                        cy=round(y + g["cy"], 1), area=g["area"], length=round(g["L"], 1),
                        width=round(g["W"], 1), elongation=round(g["elong"], 2),
                        angle=round(g["angle"], 1), peak_snr=round(peak, 1),
                        mean_snr=round(float(zc[m].mean()), 1),
                        flag=";".join(f for f in (flag, "image_edge" if edge else "") if f)))
    return inst, cls_of, cat


# --------------------------------------------------------------------- tiling
def tile_starts(n, t):
    """Tile origins covering [0, n) with tiles of size t; the last tile is shifted
    back so it ends exactly at n (overlap instead of padding / cropping)."""
    if n <= t:
        return [0]
    s = list(range(0, n - t, t))
    s.append(n - t)
    return s


def mask_to_polys(m, ox, oy, scale=4):
    """Boolean mask -> list of polygons (pixel-EDGE coordinates, tile frame).
    The mask is up-sampled so the outline follows the pixel boundary rather than
    pixel centres (a 3x3 blob keeps ~3x3 px area instead of 2x2)."""
    mp = np.pad(m, 1).astype(np.uint8)
    up = cv2.resize(mp, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        ap = cv2.approxPolyDP(c, 0.3 * scale, True).reshape(-1, 2).astype(np.float64)
        if len(ap) < 3:
            bx, by, bw, bh = cv2.boundingRect(c)
            ap = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], float)
        polys.append((ap + 0.5) / scale - 1.0 + np.array([ox, oy]))
    return polys


def tile_labels(inst_t, cls_of, min_frag):
    """YOLO-seg lines for one tile."""
    th, tw = inst_t.shape
    lines = []
    for oid, sl in enumerate(ndi.find_objects(inst_t), start=1):
        if sl is None:
            continue
        m = inst_t[sl] == oid
        if m.sum() < min_frag:
            continue
        for poly in mask_to_polys(m, sl[1].start, sl[0].start):
            xy = np.clip(poly / np.array([tw, th]), 0, 1)
            lines.append(f"{cls_of[oid]} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
    return lines


# ------------------------------------------------------------------- repatch
def repatch(tile_dir, label_dir, manifest, H, W, markers):
    """Rebuild the full image from the tile PNGs and draw polygons read from the
    YOLO label files."""
    full = np.zeros((H, W), np.uint8)
    cover = np.zeros((H, W), np.uint8)
    polys = []  # (cls, Nx2 array in full-image px)
    for t in manifest:
        im = cv2.imread(str(tile_dir / t["name"]), cv2.IMREAD_GRAYSCALE)
        full[t["y0"]:t["y0"] + t["h"], t["x0"]:t["x0"] + t["w"]] = im
        cover[t["y0"]:t["y0"] + t["h"], t["x0"]:t["x0"] + t["w"]] = 1
        lf = label_dir / (Path(t["name"]).stem + ".txt")
        for ln in lf.read_text().splitlines():
            p = ln.split()
            if len(p) < 7:
                continue
            xy = np.array(p[1:], float).reshape(-1, 2) * np.array([t["w"], t["h"]])
            polys.append((int(p[0]), xy + np.array([t["x0"], t["y0"]])))
    assert cover.all(), "tiles do not cover the full image"
    color = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    layer = color.copy()
    for k, xy in polys:
        cv2.fillPoly(layer, [np.round(xy * 16).astype(np.int32)], COLOURS[k], shift=4)
    over = cv2.addWeighted(layer, 0.65, color, 0.35, 0)
    if markers:  # small rings so 3-5 px stars are findable on a 9568 px wide image
        for k, xy in polys:
            c = xy.mean(0)
            r = int(max(7, 0.7 * (xy.max(0) - xy.min(0)).max()))
            cv2.circle(over, (int(c[0]), int(c[1])), r, COLOURS[k], 1, cv2.LINE_AA)
    return full, over


# ----------------------------------------------------------------------- main
def process_image(path, a, split):
    t0 = time.time()
    stem = Path(path).name.split(".")[0]
    raw, hdr = read_fits(path)
    if a.roi:
        x0, y0, w, h = a.roi
        raw = raw[y0:y0 + h, x0:x0 + w]
    H, W = raw.shape
    print(f"[{stem}] {W}x{H} {raw.dtype}")

    flat8, Z, info = preprocess(raw, a)
    del raw
    inst, cls_of, cat = detect(Z, a)
    del Z
    n_blob = sum(1 for c in cat if c["cls"] == "blob")
    print(f"[{stem}] bg={info['bg']:.1f} sigma={info['sigma']:.2f} hot_px={info['hot_pixels']} "
          f"-> {n_blob} blobs, {len(cat) - n_blob} streaks")

    out = Path(a.output)
    img_dir = out / "dataset" / "images" / split
    lab_dir = out / "dataset" / "labels" / split
    for d in (img_dir, lab_dir, out / "repatched", out / "overlays", out / "masks", out / "catalogues"):
        d.mkdir(parents=True, exist_ok=True)

    # tiles + labels
    manifest = []
    T = a.tile
    for r, y0 in enumerate(tile_starts(H, T)):
        for c, x0 in enumerate(tile_starts(W, T)):
            tile = flat8[y0:y0 + T, x0:x0 + T]
            th, tw = tile.shape
            lines = tile_labels(inst[y0:y0 + T, x0:x0 + T], cls_of, a.min_fragment)
            if a.skip_empty and not lines:
                continue
            name = f"{stem}_r{r:02d}_c{c:02d}.png"
            cv2.imwrite(str(img_dir / name), tile)
            (lab_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            manifest.append(dict(name=name, x0=x0, y0=y0, w=tw, h=th, n_polygons=len(lines)))
    # (skip_empty would break the coverage check, so repatch only when all tiles exist)
    (out / "catalogues" / f"{stem}_tiles.json").write_text(json.dumps(manifest, indent=1))

    # semantic mask (0 bg, 1 blob, 2 streak) for the full image
    lut = np.array([0] + [k + 1 for k in cls_of[1:]], np.uint8)
    cv2.imwrite(str(out / "masks" / f"{stem}_mask.png"), lut[inst])

    # catalogue
    with open(out / "catalogues" / f"{stem}_objects.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cat[0].keys()) if cat else ["id"])
        wr.writeheader()
        wr.writerows(cat)

    # re-patch + lossless check + overlay
    if not a.skip_empty:
        full, over = repatch(img_dir, lab_dir, manifest, H, W, not a.no_markers)
        assert np.array_equal(full, flat8), "re-patched image differs from processed image!"
        cv2.imwrite(str(out / "repatched" / f"{stem}_repatched.png"), full)
        ext = a.overlay_ext.lstrip(".")
        params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in ("jpg", "jpeg") else []
        cv2.imwrite(str(out / "overlays" / f"{stem}_overlay.{ext}"), over, params)
        print(f"[{stem}] {len(manifest)} tiles, repatched image identical to processed image")
    print(f"[{stem}] done in {time.time() - t0:.1f}s")
    return dict(image=stem, width=W, height=H, tiles=len(manifest), blobs=n_blob,
                streaks=len(cat) - n_blob, hot_pixels=info["hot_pixels"],
                bg=round(info["bg"], 1), sigma=round(info["sigma"], 2),
                review_flags=sum(1 for c in cat if "review" in c["flag"]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="FITS file or folder of FITS files")
    p.add_argument("--output", required=True)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--mesh", type=int, default=128, help="background/noise mesh box (px)")
    p.add_argument("--smooth", type=float, default=1.2, help="matched-filter Gaussian sigma (px)")
    p.add_argument("--low", type=float, default=2.0, help="hysteresis low threshold (SNR) = object extent")
    p.add_argument("--seed", type=float, default=5.5, help="hysteresis seed: peak SNR needed to accept an object")
    p.add_argument("--min-area", type=int, default=5, help="min object area (px)")
    p.add_argument("--max-area", type=int, default=20000)
    p.add_argument("--peak-frac", type=float, default=0.3, help="outline at >= this fraction of the peak SNR")
    p.add_argument("--hot-sigma", type=float, default=8.0, help="0 disables hot-pixel repair")
    p.add_argument("--streak-len", type=float, default=12.0, help="min length (px) of a streak")
    p.add_argument("--streak-elong", type=float, default=2.5, help="min length/width of a streak")
    p.add_argument("--fat-len", type=float, default=20.0, help="min length of a short/fat streak")
    p.add_argument("--fat-elong", type=float, default=1.8, help="min length/width of a short/fat streak")
    p.add_argument("--min-fragment", type=int, default=3, help="min px of an object piece inside a tile")
    p.add_argument("--stretch", type=float, nargs=2, default=[2.0, 10.0], metavar=("LO", "HI"),
                   help="8-bit stretch: bg-LO*sigma .. bg+HI*sigma")
    p.add_argument("--val-images", type=int, default=0, help="last N images go to the val split")
    p.add_argument("--skip-empty", action="store_true", help="do not write tiles without objects")
    p.add_argument("--no-markers", action="store_true", help="no rings around objects in the overlay")
    p.add_argument("--overlay-ext", default="jpg", help="jpg (small) or png")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "W", "H"),
                   help="process only this crop (for quick parameter tuning)")
    p.add_argument("--limit", type=int, help="process only the first N images")
    return p.parse_args()


def main():
    a = parse_args()
    src = Path(a.input)
    files = [src] if src.is_file() else sorted(
        f for f in src.iterdir() if f.name.lower().endswith((".fits", ".fit", ".fts", ".fits.gz")))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit("no FITS files found")
    out = Path(a.output)
    rows = []
    for i, f in enumerate(files):
        split = "val" if i >= len(files) - a.val_images else "train"
        rows.append(process_image(f, a, split))
    out.mkdir(parents=True, exist_ok=True)
    val = "images/val" if a.val_images else "images/train"
    (out / "dataset" / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: {val}\nnames:\n" +
        "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items()))
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "params.json").write_text(json.dumps(vars(a), indent=1))
    print(f"\nAll done -> {out}")


if __name__ == "__main__":
    main()
