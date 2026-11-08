# Copyright (C) 2026 Yunus Serhat Bıçakçı
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare terrain for the drone survey planner.

Reads the LiDAR for Scotland Phase 4 digital terrain model (50 cm) for three survey sites around Stirling directly
from the Scottish Remote Sensing Portal's open data bucket, averages it to a 5 m grid, fills gaps such as lake
surfaces from the surrounding ground, and writes drone_planner/terrain.js for the web page.

Run from the repository root with the lab environment:
    python tools/prepare_drone_terrain.py
"""
import base64
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.windows import from_bounds
from pyproj import Transformer

BUCKET = "https://srsp-open-data.s3.eu-west-2.amazonaws.com/lidar/phase-4/dtm/27700/gridded/"
CELL = 5  # metres

SITES = [
    {
        "id": "airthrey",
        "name": "Airthrey Loch and the university campus",
        "note": "Campus and loch on gentle ground in the south, rising steeply to the wooded Ochil escarpment in the north.",
        "tile": "NS89NW_50CM_DTM_PHASE4.tif",
        "bounds": (280150, 696250, 281350, 697050),
        "takeoff": (280300, 696330, "Grass at the south-west corner"),
    },
    {
        "id": "abbeycraig",
        "name": "Abbey Craig and the Wallace Monument",
        "note": "A steep wooded crag that rises far above the take-off point. Watch what terrain does to the survey.",
        "tile": "NS89NW_50CM_DTM_PHASE4.tif",
        "bounds": (280450, 695150, 281550, 696150),
        "takeoff": (280560, 695260, "Field at the foot of the crag"),
    },
    {
        "id": "cambuskenneth",
        "name": "River Forth floodplain at Cambuskenneth",
        "note": "Flat floodplain inside a meander of the Forth. A long, efficient survey.",
        "tile": "NS89SW_50CM_DTM_PHASE4.tif",
        "bounds": (280450, 693350, 281950, 694350),
        "takeoff": (280600, 693470, "Field by the riverbank path"),
    },
]


def read_site(site):
    e0, n0, e1, n1 = site["bounds"]
    nx, ny = int((e1 - e0) / CELL), int((n1 - n0) / CELL)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open("/vsicurl/" + BUCKET + site["tile"]) as src:
            window = from_bounds(e0, n0, e1, n1, src.transform)
            data = src.read(1, window=window, out_shape=(ny, nx), resampling=Resampling.average, masked=True)
    elevation = data.filled(np.nan).astype("float32")
    gaps = ~np.isfinite(elevation)
    if gaps.any():
        filled = fillnodata(np.where(gaps, 0, elevation), mask=(~gaps).astype("uint8"), max_search_distance=200)
        elevation = np.where(gaps, filled, elevation)
    return elevation, float(gaps.mean())


def affine_to_lonlat(bounds):
    """Fit lon and lat as linear functions of easting and northing over the site (error well under 1 m)."""
    e0, n0, e1, n1 = bounds
    to_ll = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    ee, nn = np.meshgrid(np.linspace(e0, e1, 6), np.linspace(n0, n1, 6))
    lon, lat = to_ll.transform(ee.ravel(), nn.ravel())
    design = np.c_[np.ones(ee.size), ee.ravel() - e0, nn.ravel() - n0]
    coef_lon = np.linalg.lstsq(design, lon, rcond=None)[0]
    coef_lat = np.linalg.lstsq(design, lat, rcond=None)[0]
    worst = max(np.abs(design @ coef_lon - lon).max() * 111320 * np.cos(np.radians(56.1)),
                np.abs(design @ coef_lat - lat).max() * 110574)
    return coef_lon.tolist(), coef_lat.tolist(), worst


def main():
    out = {"cell": CELL, "source": "LiDAR for Scotland Phase 4 DTM (50 cm, 2017 to 2019), averaged to 5 m, "
                                   "from the Scottish Remote Sensing Portal under the Open Government Licence. "
                                   "Crown copyright Scottish Government and Fugro (2020).", "sites": []}
    for site in SITES:
        elevation, gap_share = read_site(site)
        e0, n0, e1, n1 = site["bounds"]
        ny, nx = elevation.shape
        te, tn, tname = site["takeoff"]
        ti, tj = int((n1 - tn) / CELL), int((te - e0) / CELL)
        coef_lon, coef_lat, worst = affine_to_lonlat(site["bounds"])
        decimetres = np.round(elevation * 10).astype("<i2")
        out["sites"].append({
            "id": site["id"], "name": site["name"], "note": site["note"],
            "bounds": [e0, n0, e1, n1], "nx": nx, "ny": ny,
            "elevation_dm": base64.b64encode(decimetres.tobytes()).decode("ascii"),
            "takeoff": {"e": te, "n": tn, "name": tname, "elevation": round(float(elevation[ti, tj]), 1)},
            "lonlat": {"lon": coef_lon, "lat": coef_lat},
            "gap_share": round(gap_share, 3),
        })
        print(f"{site['id']}: {nx} x {ny} cells, elevation {np.nanmin(elevation):.1f} to {np.nanmax(elevation):.1f} m, "
              f"take-off {elevation[ti, tj]:.1f} m, gaps filled {gap_share:.1%}, lon/lat fit error {worst:.2f} m")
    target = Path(__file__).resolve().parents[1] / "drone_planner" / "terrain.js"
    target.parent.mkdir(exist_ok=True)
    target.write_text("// Generated by tools/prepare_drone_terrain.py. Do not edit by hand.\n"
                      "window.TERRAIN = " + json.dumps(out, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
