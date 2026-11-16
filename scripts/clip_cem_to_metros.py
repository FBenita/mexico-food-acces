r"""
clip_cem_to_metros.py

Cuts the national elevation raster of INEGI (Continuo de Elevaciones Mexicano
4.0, 15 m) into one GeoTIFF per metropolitan area, which notebook 13 reads.

Run with:  python clip_cem_to_metros.py --base <project folder> --cem <folder or file holding the CEM>

What it does
------------
1. Finds the CEM GeoTIFF and prints its diagnostics (CRS, tiling, dtype,
   NoData).
2. Reads the 92 metro GeoPackages, dissolves each one (metros spanning several
   municipalities arrive as several polygons), buffers by 2 km so that network
   edges running to the metro edge still have elevation on both ends, and
   reprojects the buffered bounds into the raster's own CRS.
3. Writes one clipped, tiled, DEFLATE-compressed GeoTIFF per metro.
4. Writes a log CSV with per-metro diagnostics.

Design notes
------------
* Reading is windowed and chunked by rows, so peak memory is a few tens of MB
  regardless of metro size.
* The DEM is NOT reprojected. Resampling elevations introduces error for no
  benefit; sample points get reprojected into the DEM's CRS instead, which is
  lossless.
* Bounding box rather than polygon mask, on purpose. Network nodes can sit just
  outside the dissolved boundary, and a polygon mask would hand them NoData.
  Empty corners cost almost nothing once DEFLATE has run over them.
* Re-running skips metros already done. Output is written to a .tmp file and
  renamed only on success, so an interrupted run never leaves a half file that
  a later run would mistake for finished.

Requirements
------------
pip install rasterio geopandas shapely

The output, about 1 GB across 92 files, goes to <project>/Inputs/DEM.
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
import time
import csv
import zipfile

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
_ap.add_argument("--base", required=True, help="project folder holding Inputs/")
_ap.add_argument("--cem", required=True,
                 help="the CEM GeoTIFF, a folder holding it, or its zip archive "
                      "(an extracted raster reads much faster)")
_args = _ap.parse_args()

BASE = _args.base
CEM_SOURCE = _args.cem

METROS_DIR = os.path.join(BASE, "Inputs", "metro_areas")
OUT_DIR = os.path.join(BASE, "Inputs", "DEM")
LOG_CSV = os.path.join(OUT_DIR, "_cem_clip_log.csv")

BUFFER_M = 2000          # metres of margin around each dissolved metro
METRIC_CRS = 6372        # ITRF2008 / LCC, for buffering in real metres
CHUNK_ROWS = 2048        # rows read at a time; lower this if memory is tight

# ---------------------------------------------------------------------------

try:
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window, from_bounds, intersection
    from rasterio.warp import transform_bounds
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e}\n\n    pip install rasterio geopandas shapely\n")


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def dissolve(gdf):
    """Union a GeoDataFrame's geometries, across geopandas versions."""
    geoms = gdf.geometry.buffer(0)
    try:
        return geoms.union_all()
    except AttributeError:
        return geoms.unary_union


# ---------------------------------------------------------------------------
# 1. Locate the raster
# ---------------------------------------------------------------------------

def find_dem(source: str) -> str:
    """Return a rasterio-openable path for the CEM, extracted or inside a zip."""
    if os.path.isfile(source) and source.lower().endswith(".tif"):
        return source

    if os.path.isdir(source):
        tifs = sorted(glob.glob(os.path.join(source, "**", "*.tif"), recursive=True))
        if tifs:
            tifs.sort(key=os.path.getsize, reverse=True)
            return tifs[0]
        zips = sorted(glob.glob(os.path.join(source, "**", "*.zip"), recursive=True))
        if zips:
            source = zips[0]

    if os.path.isfile(source) and source.lower().endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        if not inner:
            sys.exit(f"No .tif inside {source}")
        print("\n*** Reading directly from the zip. This works but is MUCH slower.")
        print("*** Extracting the archive first is strongly recommended.\n")
        vpath = source.replace("\\", "/")
        return f"zip://{vpath}!{inner[0]}"

    sys.exit(f"Could not find a CEM GeoTIFF under: {source}")


def diagnose(path: str) -> dict:
    with rasterio.open(path) as src:
        blocks = src.block_shapes[0] if src.block_shapes else (None, None)
        tiled = bool(src.profile.get("tiled", False))
        info = {
            "path": path,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "tiled": tiled,
            "block": f"{blocks[1]}x{blocks[0]}",
            "res": src.res,
            "bounds": src.bounds,
        }

    banner("RASTER DIAGNOSTICS")
    for k in ("path", "crs", "width", "height", "dtype", "nodata", "tiled", "block", "res"):
        print(f"  {k:<8s}: {info[k]}")
    print(f"  bounds  : {info['bounds']}")
    cells = info["width"] * info["height"]
    print(f"  cells   : {cells:,}")
    if not info["tiled"]:
        print("\n  NOTE: raster is STRIPED, not tiled. Window reads will pull full")
        print("        scanlines, so expect this job to be slow but not heavy.")
    return info


# ---------------------------------------------------------------------------
# 2. Metro bounds, dissolved and buffered, in the raster's CRS
# ---------------------------------------------------------------------------

def metro_bounds(dem_crs) -> dict:
    files = sorted(glob.glob(os.path.join(METROS_DIR, "metro_*.gpkg")))
    if not files:
        sys.exit(f"No .gpkg files found in {METROS_DIR}")
    print(f"\nmetro boundary files found: {len(files)}")

    out = {}
    for f in files:
        key = os.path.splitext(os.path.basename(f))[0]
        g = gpd.read_file(f)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if g.empty:
            print(f"  WARNING: {key} has no usable geometry, skipped")
            continue
        if g.crs is None:
            g = g.set_crs(4326)

        poly = dissolve(g.to_crs(epsg=METRIC_CRS)).buffer(BUFFER_M)
        left, bottom, right, top = gpd.GeoSeries([poly], crs=f"EPSG:{METRIC_CRS}").total_bounds
        out[key] = transform_bounds(f"EPSG:{METRIC_CRS}", dem_crs,
                                    left, bottom, right, top, densify_pts=21)

    print(f"dissolved and buffered: {len(out)} metros")
    return out


# ---------------------------------------------------------------------------
# 3. Clip
# ---------------------------------------------------------------------------

def clip_one(src, bounds, out_path: str) -> dict:
    win = from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    try:
        win = intersection(win, Window(0, 0, src.width, src.height))
    except Exception:
        return {"status": "outside_raster", "width": 0, "height": 0, "mb": 0.0}
    if win.width <= 0 or win.height <= 0:
        return {"status": "outside_raster", "width": 0, "height": 0, "mb": 0.0}

    w, h = int(win.width), int(win.height)
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        transform=src.window_transform(win),
        compress="deflate",
        predictor=3 if src.dtypes[0].startswith("float") else 2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    profile.pop("nbits", None)

    tmp = out_path + ".tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        for row in range(0, h, CHUNK_ROWS):
            n = min(CHUNK_ROWS, h - row)
            sub = Window(win.col_off, win.row_off + row, w, n)
            dst.write(src.read(1, window=sub), 1, window=Window(0, row, w, n))

    os.replace(tmp, out_path)
    return {"status": "ok", "width": w, "height": h,
            "mb": round(os.path.getsize(out_path) / 1e6, 2)}


# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    dem_path = find_dem(CEM_SOURCE)
    info = diagnose(dem_path)
    bounds_by_metro = metro_bounds(info["crs"])

    banner(f"CLIPPING {len(bounds_by_metro)} METROS  ->  {OUT_DIR}")
    rows, t_start = [], time.time()

    with rasterio.open(dem_path) as src:
        for i, (key, bounds) in enumerate(sorted(bounds_by_metro.items()), 1):
            out_path = os.path.join(OUT_DIR, f"{key}.tif")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"[{i:>2}/{len(bounds_by_metro)}] {key:<34s} already done")
                rows.append({"metro_key": key, "status": "cached",
                             "width": "", "height": "", "mb": "", "seconds": ""})
                continue

            t0 = time.time()
            try:
                res = clip_one(src, bounds, out_path)
            except Exception as exc:
                for junk in (out_path + ".tmp",):
                    if os.path.exists(junk):
                        os.remove(junk)
                print(f"[{i:>2}/{len(bounds_by_metro)}] {key:<34s} FAILED: {exc}")
                rows.append({"metro_key": key, "status": f"error: {exc}",
                             "width": "", "height": "", "mb": "", "seconds": ""})
                continue

            secs = round(time.time() - t0, 1)
            print(f"[{i:>2}/{len(bounds_by_metro)}] {key:<34s} "
                  f"{res['width']:>6,} x {res['height']:>6,} px  "
                  f"{res['mb']:>7.2f} MB  {secs:>6.1f}s  {res['status']}")
            rows.append({"metro_key": key, **res, "seconds": secs})

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["metro_key", "status", "width", "height", "mb", "seconds"])
        writer.writeheader()
        writer.writerows(rows)

    done = [r for r in rows if r["status"] in ("ok", "cached")]
    total_mb = sum(float(r["mb"]) for r in rows if r["status"] == "ok")
    banner("DONE")
    print(f"  metros written or cached : {len(done)} / {len(rows)}")
    print(f"  new output size          : {total_mb:,.0f} MB")
    print(f"  elapsed                  : {(time.time() - t_start) / 60:.1f} min")
    print(f"  log                      : {LOG_CSV}")
    failed = [r['metro_key'] for r in rows if r["status"] not in ("ok", "cached")]
    if failed:
        print(f"\n  NOT DONE ({len(failed)}): {', '.join(failed)}")
        print("  Re-run the script; finished metros are skipped.")


if __name__ == "__main__":
    main()
