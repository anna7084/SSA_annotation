# Question 2 - 

**Method.** Script-based (custom Python pipeline, repository: [https://github.com/anna7084/SSA_annotation.git]. Detection runs
on the full 9568 x 6380 frame, then the result is cut into tiles. Class 0 = blob (star), class 1 = streak.

## a) Pre-processing

1. **Inspection.** Each FITS file read as a 2-D uint16 array; background level, robust noise
   sigma (1.4826 x MAD) and saturation fraction checked (`inspect_fits.py`).
2. **Local background and noise map.** 128 x 128 px mesh, 3-sigma iterative clipping per box,
   median-filtered across boxes and bilinearly up-sampled. This follows sky gradients and is not
   biased by stars because they are clipped.
3. **Hot-pixel repair.** A pixel is replaced by the 3x3 median when it is > 8 sigma above the
   local background *and* all eight neighbours are < 2.5 sigma - a real star always has a
   neighbourhood, a hot pixel does not.
4. **Flattening and stretch.** Background subtracted (global sky level re-added), then one linear
   stretch per image, (bg - 2 sigma) .. (bg + 10 sigma), to 8 bit. One scaling per image, so all
   tiles of an image share the same brightness mapping.
5. **Detection image.** Gaussian smoothing (sigma 1.2 px, roughly matched to a star PSF) and
   conversion to a per-pixel SNR map. Detection thresholds therefore have a statistical meaning.

*Assumption:* noise is approximately Gaussian and stationary on 128 px scales.

## b) Tile boundaries without data loss

Tile origins per axis: `range(0, N - 1024, 1024)` followed by `N - 1024`.
For 9568 this is 0, 1024, ..., 8192, 8544; for 6380 it is 0, ..., 5120, 5356, giving
10 x 7 = 70 full-size 1024 x 1024 tiles. The last tile in each direction is shifted back so it ends
exactly on the image edge (overlap of 672 px horizontally and 788 px vertically with its
neighbour) instead of being padded (invented pixels) or cropped (lost pixels).

Objects are detected on the full frame, so an object crossing a tile edge keeps one class and one
outline; each tile receives its clipped piece (pieces < 3 px are dropped). Verification: the
script re-patches the tiles by their offsets and asserts that the result is **bit-identical** to the
processed full-size image; the overlay is drawn from the exported label files.

## c) Short/fat streaks versus blobs

For every object the minimum-area rectangle of its mask gives length L and width W (pixels, +1 for
pixel extent); elongation = L / W.

* **Streak:** L >= 12 px and L/W >= 2.5 (normal streaks), **or** L >= 20 px and L/W >= 1.8
  (short/fat streaks - a wide, short trail is still much longer than the PSF).
* **Blob:** everything else. A star is round, so L/W is near 1 and L is a few pixels.
* **Review flags** in `catalogues/*_objects.csv`: blobs with L/W >= 1.6 and L >= 8, and streaks
  with L < 15.6 px, were re-checked by eye in the overlay.

*Assumption:* a star's trail is shorter than ~12 px because exposures are short.

## d) Faint / small blobs as the star class

* **Seed:** the peak of the smoothed SNR map must reach **5.5 sigma**. Measured on a pure-noise
  frame: 4.5 sigma gave ~240 false detections per full image, 5.5 sigma gave 0.
* **Extent:** the object grows to all connected pixels with SNR >= 2 (hysteresis), then the outline
  is cut at 30 % of the object's own peak so bright stars do not get a bloated halo while faint ones
  keep their full footprint.
* **Minimum size:** >= 5 px, which rejects single hot pixels and noise spikes; maximum 20 000 px
  (rejects large diffuse residuals).
* Faint objects meeting these criteria are labelled class 0 regardless of brightness.
  Objects below 5.5 sigma are **not** annotated (treated as background) - this is a deliberate
  trade-off: missing the faintest stars is preferred to training on noise.

## Deliverables

`output/` as described in the README: tiles + YOLO-seg labels, full-size re-patched images and
overlays, semantic masks, catalogues. Run instructions: README.md.
