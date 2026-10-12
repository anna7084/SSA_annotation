#!/usr/bin/env python3
"""inspect_fits.py - quick look at one FITS image (stats + downsampled preview).

  python inspect_fits.py image.fits [preview.png]
"""
import sys

import cv2
import numpy as np
from astropy.io import fits

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "preview.png"

with fits.open(path, memmap=False) as hdul:
    hdul.info()
    hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
    data = np.asarray(hdu.data)
    while data.ndim > 2:
        data = data[0]
    hdr = hdu.header

print("\nshape (rows, cols):", data.shape, "dtype:", data.dtype, "BITPIX:", hdr.get("BITPIX"))
sub = data[::4, ::4].astype(np.float32)
med = float(np.median(sub))
sig = 1.4826 * float(np.median(np.abs(sub - med)))
print(f"min={data.min()} max={data.max()} median={med:.1f} robust_sigma={sig:.2f}")
print(f"saturated (>=65535): {(data >= 65535).mean() * 100:.4f}%   zeros: {(data == 0).mean() * 100:.4f}%")
for k in ("EXPTIME", "EXPOSURE", "DATE-OBS", "TELESCOP", "INSTRUME", "FILTER", "GAIN", "BZERO", "BSCALE"):
    if k in hdr:
        print(f"{k:9s}= {hdr[k]}")

lo, hi = med - 2 * sig, med + 10 * sig
small = cv2.resize(np.clip((data.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8),
                   None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
cv2.imwrite(out, small)
print("preview ->", out)
