"""
build_dataset.py: Data preprocessing for Orange Madagascar Dashboard.

Replicates merge logic from data_exploration.ipynb:
- Load 4 raw data (user locations, forfait purchases, sites reference, cell info)
- Join sites to cells, then users to enriched locations to forfaits
- Apply forfait pricing to create CA column
- Clean region names for GeoJSON matching
- Export df_merged.parquet and madagascar_regions.geojson
"""

import sys
#!pip install XXX --target ./my_custom_packages #openpyxl
sys.path.append('./my_custom_packages')  

import polars as pl
import pandas as pd
import numpy as np
import json
import os


PROJECT = os.path.dirname(os.path.abspath(__file__))

# Input files 
LOC_PATH     = os.path.join(PROJECT, "loc_quarter.parquet")
FORFAIT_PATH = os.path.join(PROJECT, "forfait_quarter.parquet")
SITES_PATH   = os.path.join(PROJECT, "REF_sites_V4.xlsx")
SIGCELL_PATH = os.path.join(PROJECT, "rf_sig_cell_v3.parquet")
SHP_PATH     = os.path.join(PROJECT, "MDG_adm", "MDG_adm2.shp") 

# Output files
OUT_MERGED   = os.path.join(PROJECT, "df_merged.parquet")
OUT_GEOJSON  = os.path.join(PROJECT, "madagascar_regions.geojson")


def clean_region_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean sig_region_name to match GeoJSON NAME_2 field.
    """
    return     df.with_columns(
        pl.when(pl.col("sig_region_name").str.contains("(?i)^atsinanana$"))
        .then(pl.lit("Atsinanana"))
        .when(pl.col("sig_region_name").str.contains("(?i)vatovavy|fitovinany"))
        .then(pl.lit("Vatovavy Fitovinany"))
        .when(pl.col("sig_region_name").str.contains("(?i)atsinanana"))
        .then(pl.lit("Atsimo-Atsinana"))
        .when(pl.col("sig_region_name").str.contains("(?i)andrefana"))
        .then(pl.lit("Atsimo-Andrefana"))
        .when(pl.col("sig_region_name").str.contains("(?i)mangoro"))
        .then(pl.lit("Alaotra-Mangoro"))
        .when(pl.col("sig_region_name").str.contains("(?i)fitovinany"))
        .then(pl.lit("Vatovavy Fitovinany"))
        .when(pl.col("sig_region_name").str.contains("(?i)mania"))
        .then(pl.lit("Amoron'i mania"))
        .when(pl.col("sig_region_name").str.contains("vambony"))
        .then(pl.lit("Haute matsiatra"))
        .when(pl.col("sig_region_name").str.contains("(?i)matsiatra"))
        .then(pl.lit("Haute matsiatra"))
        .when(pl.col("sig_region_name").str.contains("(?i)vatovavy"))
        .then(pl.lit("Vatovavy Fitovinany"))
        .when(pl.col("sig_region_name").str.contains("(?i)vatovavy"))
        .then(pl.lit("Vatovavy Fitovinany"))

        .otherwise(pl.col("sig_region_name").str.to_titlecase())
        .alias("region_cleaned")
    )



def build_geojson():
    """
    Load and simplify the Madagascar boundaries shapefile.
    """
    import geopandas as gpd
    import shapely

    gdf = gpd.read_file(SHP_PATH)

    # Simplify geometries for faster rendering 0.01° ≈ 1km tolerance
    gdf["geometry"] = shapely.simplify(
        gdf["geometry"].values, tolerance = 0.01, preserve_topology = True
    )

    # Keep only needed columns
    gdf = gdf[["NAME_1", "NAME_2", "geometry"]]
    gdf = gdf.rename(columns = {"NAME_1": "province", "NAME_2": "region"})

    # Save as GeoJSON
    gdf.to_file(OUT_GEOJSON, driver = "GeoJSON")
    
    return gdf


def build_merged():
    """
    Merge all data sources into df_merged.parquet.
    """

    # Load data
    
    loc_mai = pl.read_parquet(LOC_PATH)

    forfait_mai = pl.read_parquet(FORFAIT_PATH)
    forfait_mai = forfait_mai.with_columns(pl.col("msisdn").cast(pl.String))

    sites = pl.from_pandas(pd.read_excel(SITES_PATH))

    sig_cell = pl.read_parquet(SIGCELL_PATH)

    partial_loc_info = sites.join(
        sig_cell, left_on = "code_site GMT", right_on = "sig_code_site", how = "left"
    )

    full_loc_info = loc_mai.join(
        partial_loc_info, left_on = "site_id", right_on = "sig_lac_ci", how = "inner"
    )

    df_merged = full_loc_info.join(forfait_mai, on ="msisdn", how = "inner")

    # Cast types
    df_merged = df_merged.with_columns([
        pl.col("sig_x").cast(pl.Float64, strict = False).alias("x"),
        pl.col("sig_y").cast(pl.Float64, strict = False).alias("y"),
        pl.col("date").cast(pl.Date, strict = False),
        pl.col("CA").cast(pl.Float64, strict = False)
    ])


    # Filter nulls and select columns

    df_merged = df_merged.filter(pl.col("Group_canal").is_not_null())

    cols_to_select = [
        "msisdn", "site_id", "nb_jours",
        "date", "Gamme_groupe", "Nom du forfait", "CA", "Group_canal",
        "sig_region_name", "sig_district_name", "sig_commune_name",
        "x", "y", "Max_RAT", "Typologie", "Typologie site", "sig_nom_site"
    ]

    df_merged = df_merged.select(cols_to_select)

    # Clean region names
    df_merged = df_merged.filter(pl.col("sig_region_name") != "")
    df_merged = clean_region_names(df_merged)

    print(f" Final dataset: {df_merged.shape}")
    print(f" Columns: {df_merged.columns}")
    print(f" Unique users: {df_merged['msisdn'].n_unique()}")
    print(f" Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    print(f" Regions: {df_merged['region_cleaned'].n_unique()}")
    print(f" Technologies: {sorted(df_merged['Max_RAT'].unique().to_list())}")

    # Save
    df_merged.write_parquet(OUT_MERGED)

    return df_merged


if __name__ == "__main__":

    # Build the merged dataset
    df = build_merged()

    # Build the GeoJSON
    build_geojson()

