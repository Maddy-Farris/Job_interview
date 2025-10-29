#!/usr/bin/env python3
"""
compute_property_orientation.py

Determine which direction each property faces based on the nearest road.

Requires:
    pip install pandas geopandas shapely pyproj pyarrow

Input files:
    - gnaf_prop.parquet  (properties, with latitude/longitude)
    - roads.gpkg         (road line geometries)

Output:
    - property_orientations.csv
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from math import atan2, degrees
from pyproj import CRS
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------- CONFIG -------------------
PROP_FN = "gnaf_prop.parquet"
ROADS_FN = "roads.gpkg"
OUT_CSV = "property_orientations.csv"
DEFAULT_CRS = "EPSG:7844"  # GDA2020, change if needed
# ----------------------------------------------

def bearing_from_vector(dx, dy):
    ang = atan2(dx, dy)
    deg = (degrees(ang) + 360) % 360
    return deg

def degree_to_cardinal(deg):
    dirs = [
        (0, 22.5, "North"),
        (22.5, 67.5, "North-East"),
        (67.5, 112.5, "East"),
        (112.5, 157.5, "South-East"),
        (157.5, 202.5, "South"),
        (202.5, 247.5, "South-West"),
        (247.5, 292.5, "West"),
        (292.5, 337.5, "North-West"),
        (337.5, 360, "North"),
    ]
    for lo, hi, name in dirs:
        if lo <= deg < hi:
            return name
    return "Unknown"

def choose_metric_crs(gdf):
    if gdf.crs is None:
        raise ValueError("Missing CRS")
    crs = CRS(gdf.crs)
    if crs.is_projected:
        return crs
    lon, lat = gdf.unary_union.centroid.x, gdf.unary_union.centroid.y
    if 112 <= lon <= 154 and -44 <= lat <= -10:
        return CRS.from_epsg(3577)
    zone = int((lon + 180) / 6) + 1
    return CRS.from_epsg(32700 + zone if lat < 0 else 32600 + zone)

# ------------------- LOAD AND FLATTEN PARQUET -------------------
print("Loading property data...")
df = pd.read_parquet(PROP_FN)

# Flatten MultiIndex if present
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index(level=list(range(df.index.nlevels)))

# Identify latitude/longitude columns
lon_candidates = [c for c in df.columns if c.lower() in ["lon", "longitude", "x"]]
lat_candidates = [c for c in df.columns if c.lower() in ["lat", "latitude", "y"]]
if not lon_candidates or not lat_candidates:
    raise ValueError("Could not find latitude/longitude columns in property dataset")
lon_col, lat_col = lon_candidates[0], lat_candidates[0]

# Create GeoDataFrame
gdf_props = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs=DEFAULT_CRS
)
gdf_props["prop_id"] = gdf_props.index

# ------------------- LOAD ROADS -------------------
print("Loading roads...")
roads = gpd.read_file(ROADS_FN)
if roads.crs is None:
    roads.set_crs(DEFAULT_CRS, inplace=True)
roads = roads.reset_index(drop=True)
roads["road_id"] = roads.index

# ------------------- REPROJECT TO METRIC -------------------
metric_crs = choose_metric_crs(gdf_props)
print(f"Reprojecting to metric CRS: {metric_crs}")
gdf_props = gdf_props.to_crs(metric_crs)
roads = roads.to_crs(metric_crs)

# ------------------- SPATIAL JOIN: NEAREST ROAD -------------------
print("Finding nearest road for each property...")
joined = gpd.sjoin_nearest(
    gdf_props, roads, how="left", distance_col="dist_m"
)

# Flatten MultiIndex columns if present
if isinstance(joined.columns, pd.MultiIndex):
    joined.columns = [col[1] if col[1] else col[0] for col in joined.columns]

# Drop index_right if exists
if "index_right" in joined.columns:
    joined = joined.drop(columns=["index_right"])

# ------------------- COMPUTE ORIENTATION -------------------
print("Computing orientation...")
results = []
for _, row in joined.iterrows():
    prop_geom = row.geometry
    road_id = row.get("road_id")

    if pd.isna(road_id):
        results.append({
            "prop_id": row["prop_id"],
            "orientation_deg": None,
            "orientation": None,
            "dist_m": None
        })
        continue

    road_geom = roads.loc[roads["road_id"] == road_id, "geometry"].values[0]
    nearest_point = road_geom.interpolate(road_geom.project(prop_geom))
    dx, dy = nearest_point.x - prop_geom.x, nearest_point.y - prop_geom.y
    bearing = bearing_from_vector(dx, dy)
    orientation = degree_to_cardinal(bearing)

    results.append({
        "prop_id": row["prop_id"],
        "orientation_deg": bearing,
        "orientation": orientation,
        "dist_m": row["dist_m"]
    })

res_df = pd.DataFrame(results)

# ------------------- COMPILE FINAL OUTPUT -------------------
out = res_df.copy()

# Safely merge address column if present
address_col = None
for c in ["address", "ADDRESS", "full_address", "FULL_ADDRESS"]:
    if c in joined.columns:
        address_col = c
        break

if address_col:
    out = out.merge(joined[["prop_id", address_col]], on="prop_id", how="left")

# Define final columns
keep_cols = ["prop_id"]
if address_col:
    keep_cols.append(address_col)
keep_cols += ["orientation", "orientation_deg", "dist_m"]

print(f"Writing results to {OUT_CSV} ...")
out[keep_cols].to_csv(OUT_CSV, index=False)

print("✅ Done! Saved orientations to", OUT_CSV)
print(out[keep_cols].head())
