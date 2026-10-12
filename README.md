# SSA star / streak annotation pipeline

Automatic pixel-level annotation of single-band 16-bit FITS space-imagery:

| class id | name   | what it is                          |
|----------|--------|-------------------------------------|
| 0        | blob   | stars / point-like spots            |
| 1        | streak | satellites / debris (line-like)     |

Output is YOLO-Ultralytics **segmentation** format on 1024 x 1024 tiles, plus the
re-patched full-size images with the labels drawn on top for visual inspection.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 0) (optional) look at one raw frame: statistics + a 4x down-sampled preview
python inspect_fits.py Datasets_Assessment/image_01.fits preview.png

# 1) tune on a single 1024x1024 crop first (fast, ~2 s)
python annotate_ssa.py --input Datasets_Assessment/image_01.fits --output tune \
       --roi 3000 2000 1024 1024
#    open tune/overlays/*.jpg, adjust --seed / --low / --peak-frac / --streak-len ... if needed

# 2) full run on all 10 images (~30 s and ~3 GB RAM per image)
python annotate_ssa.py --input Datasets_Assessment --output output
```

## Output layout

```
output/
  dataset/
    data.yaml                    Ultralytics dataset file (0: blob, 1: streak)
    images/train/*.png           1024x1024 8-bit tiles   (<image>_r<row>_c<col>.png)
    labels/train/*.txt           YOLO segmentation polygons, one object per line
  repatched/<image>_repatched.png   tiles stitched back to full size (verified identical
                                    to the processed image, the script asserts this)
  overlays/<image>_overlay.jpg      full-size image + coloured masks read back from the
                                    label files (blob = purple, streak = red)
  masks/<image>_mask.png            full-size semantic mask: 0 background, 1 blob, 2 streak
  catalogues/<image>_objects.csv    per-object geometry/SNR/class + review flags
  catalogues/<image>_tiles.json     tile offsets
  summary.csv, params.json          counts and the exact parameters used
```

Check the result with Ultralytics, e.g. `yolo segment train data=output/dataset/data.yaml model=yolov8n-seg.pt imgsz=1024`.

## How it works (short)

1. **Pre-processing** - 128 px mesh of sigma-clipped background + noise, hot-pixel repair,
   background flattening, one global linear stretch per image (bg-2sigma .. bg+10sigma) to 8 bit.
2. **Detection on the full frame** (never on tiles, so no object is split or classified
   differently at a tile border) - Gaussian matched filter -> SNR map -> hysteresis
   (seed SNR >= 5.5, extent SNR >= 2) -> per-object outline at >= 30 % of its peak.
3. **Classification** - length/width from the min-area rectangle: streak if L >= 12 px and
   L/W >= 2.5, or (short/fat) L >= 20 px and L/W >= 1.8; everything else is a blob.
   Borderline objects get a `*_review` flag in the catalogue.
4. **Tiling** - origins `0, 1024, ...` plus a last origin at `size - 1024`, so edge tiles are
   full-size and overlap their neighbour instead of being padded or cropped
   (9568 x 6380 -> 10 x 7 = 70 tiles per image).
5. **Labels** - polygons of every object piece inside a tile (pieces < 3 px dropped),
   normalised to the tile size.

All thresholds are command-line options (`python annotate_ssa.py -h`).

## Notes / limitations

* Defaults were validated on a synthetic frame with the same size, dtype and noise character
  as the assessment images, **not** on the real data - tune on a crop of a real image first.
* Streaks broken into several separate pieces by noise (very faint, beaded) are not
  re-linked; crowded stars closer than ~2 px merge into one blob.
* Without `astropy` the script falls back to a minimal FITS reader (single image, BITPIX 16);
  install astropy for anything else.
