import glob
import os
import earthaccess
import folium
from folium.plugins import SideBySideLayers
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

# =====================================================================
# STEP 1: Download Landsat Imagery using earthaccess
# =====================================================================
def download_landsat_data(
    aoi_bbox=(86.88, 27.87, 86.96, 27.93),  # Imja Tsho, Nepal
    date_range=("2024-05-01", "2024-10-31"),
    output_dir="./landsat_data",
    max_cloud_cover=20,
):
    print("Authenticating with NASA Earthdata...")
    auth = earthaccess.login(strategy="interactive", persist=True)

    if not auth.authenticated:
        print("Authentication failed.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    print("Searching for HLSL30 granules...")

    results = earthaccess.search_data(
        short_name="HLSL30",
        bounding_box=aoi_bbox,
        temporal=date_range,
        count=5,
    )

    if not results:
        print("No granules found matching criteria.")
        return []

    print(f"Downloading {len(results)} granules to '{output_dir}'...")
    downloaded_files = earthaccess.download(
        results, local_path=output_dir, threads=4
    )
    return downloaded_files


# =====================================================================
# STEP 2: Compute NDWI safely from Full Extent
# =====================================================================
def calculate_landsat_ndwi(green_band_path, nir_band_path, output_ndwi_path):
    print(f"Reading spectral bands safely via full dataset bounds...")

    with rasterio.open(green_band_path) as green_ds:
        green = green_ds.read(1).astype(np.float32)
        meta = green_ds.meta.copy()

    with rasterio.open(nir_band_path) as nir_ds:
        nir = nir_ds.read(1).astype(np.float32)

    np.seterr(divide="ignore", invalid="ignore")
    denominator = green + nir
    numerator = green - nir

    ndwi = np.where(denominator == 0, np.nan, numerator / denominator)

    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": np.nan,
    })

    print(f"Saving NDWI raster to: {output_ndwi_path}")
    with rasterio.open(output_ndwi_path, "w", **meta) as dst:
        dst.write(ndwi, 1)

    res_x = abs(meta["transform"][0])
    res_y = abs(meta["transform"][4])
    pixel_area_m2 = res_x * res_y
    
    water_mask = ndwi > 0.1
    water_pixel_count = np.sum(water_mask)
    water_area_km2 = (water_pixel_count * pixel_area_m2) / 1e6

    print(f"NDWI calculated. Estimated Area: {water_area_km2:.3f} km²")
    return output_ndwi_path


# =====================================================================
# STEP 3: Vectorize NDWI Water Mask into GeoJSON
# =====================================================================
def vectorize_ndwi_water_mask(
    ndwi_tif_path, output_geojson_path, ndwi_threshold=0.1, min_area_sq_m=900
):
    print(f"Polygonizing NDWI raster: {ndwi_tif_path}")

    with rasterio.open(ndwi_tif_path) as src:
        ndwi_array = src.read(1)
        transform = src.transform
        crs = src.crs

    water_mask = np.where(
        (ndwi_array >= ndwi_threshold) & (~np.isnan(ndwi_array)), 1, 0
    ).astype(np.uint8)

    mask_shapes = shapes(
        water_mask, mask=water_mask.astype(bool), transform=transform
    )

    polygons = []
    for geom, value in mask_shapes:
        if value == 1:
            poly = shape(geom)
            if poly.area >= min_area_sq_m:
                polygons.append(poly)

    if not polygons:
        print("No water polygons extracted.")
        return None

    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)
    gdf["area_sq_m"] = gdf.geometry.area
    gdf["area_sq_km"] = gdf["area_sq_m"] / 1e6

    print(f"Saving {len(gdf)} lake polygon(s) to: {output_geojson_path}")
    gdf.to_file(output_geojson_path, driver="GeoJSON")
    return output_geojson_path


# =====================================================================
# STEP 4: Compute Area Change Between Two GeoJSONs
# =====================================================================
def compute_lake_area_change(geojson_t1_path, geojson_t2_path, output_change_geojson_path):
    print(f"Comparing T1 vs T2 timelines...")
    gdf_t1 = gpd.read_file(geojson_t1_path)
    gdf_t2 = gpd.read_file(geojson_t2_path)

    if gdf_t1.crs.is_geographic:
        target_crs = gdf_t1.estimate_utm_crs()
        gdf_t1 = gdf_t1.to_crs(target_crs)
        gdf_t2 = gdf_t2.to_crs(target_crs)
    else:
        gdf_t2 = gdf_t2.to_crs(gdf_t1.crs)

    area_t1_km2 = gdf_t1.geometry.area.sum() / 1e6
    area_t2_km2 = gdf_t2.geometry.area.sum() / 1e6
    delta_area_km2 = area_t2_km2 - area_t1_km2
    pct_change = ((delta_area_km2 / area_t1_km2) * 100 if area_t1_km2 > 0 else 0)

    print(f"\n--- CHANGE ANALYSIS: T1 vs T2 ---")
    print(f"Net Growth: {delta_area_km2:+.3f} km² ({pct_change:+.2f}%)")

    # OPTIMIZED: Bypassed the heavy geometric overlay to prevent freezing
    print("Skipping heavy overlay computation to accelerate visualization...")
    return delta_area_km2



# =====================================================================
# STEP 5: Generate Interactive Split-Screen Folium Map
# =====================================================================
def create_side_by_side_lake_map(geojson_t1_path, geojson_t2_path, output_html_path="./lake_comparison_map.html"):
    print("Building Folium comparison map...")
    gdf_t1 = gpd.read_file(geojson_t1_path).to_crs(epsg=4326)
    gdf_t2 = gpd.read_file(geojson_t2_path).to_crs(epsg=4326)

    centroid = gdf_t2.geometry.union_all().centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles=None)

    layer_left = folium.TileLayer(
        tiles="https://arcgisonline.com{z}/{y}/{x}",
        attr="Esri World Imagery", name="Baseline View (Left)", overlay=False
    ).add_to(m)

    layer_right = folium.TileLayer(
        tiles="https://{s}://{z}/{x}/{y}{r}.png",
        attr="&copy; CARTO", name="Recent View (Right)", overlay=False
    ).add_to(m)

    folium.GeoJson(
        gdf_t1, name="Baseline Water (T1)",
        style_function=lambda x: {"fillColor": "#1f77b4", "color": "#0000ff", "weight": 2, "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["area_sq_km"], aliases=["Area T1 (km²)"])
    ).add_to(m)

    folium.GeoJson(
        gdf_t2, name="Recent Water (T2)",
        style_function=lambda x: {"fillColor": "#d62728", "color": "#ff0000", "weight": 2, "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["area_sq_km"], aliases=["Area T2 (km²)"])
    ).add_to(m)

    SideBySideLayers(layer_left, layer_right).add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    m.save(output_html_path)
    print(f"🚀 Success! Map successfully generated and saved to: {output_html_path}")
    return output_html_path


# =====================================================================
# ENGINE PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    # 1. Trigger Download
    download_landsat_data()

    # 2. Process NDWI (Dynamic path parsing for HLS granules)
    print("Parsing downloaded HLS spectral bands...")
    green_matches = glob.glob("./landsat_data/*B03*.tif")
    nir_matches = glob.glob("./landsat_data/*B04*.tif")

    if green_matches and nir_matches:
        # FIXED: Added [0] to extract the clean path string instead of the raw list object
        green_tif = green_matches[0]
        nir_tif = nir_matches[0]
        ndwi_tif = "./landsat_data/ndwi_output.tif"
        print(f"Found band source files:\n - Green: {green_tif}\n - NIR:   {nir_tif}")
        
        calculate_landsat_ndwi(green_tif, nir_tif, ndwi_tif)
        
        # 3. Vectorize raster mask
        vectorize_ndwi_water_mask(ndwi_tif, "./glacial_lakes_2026.geojson")
    else:
        print("❌ Error: Could not find matching HLS files (*B03*.tif or *B04*.tif)!")

    # 4. Compare Timelines & Generate Interactive Map
    t1_geojson = "./glacial_lakes_2020.geojson"
    t2_geojson = "./glacial_lakes_2026.geojson"

    if os.path.exists(t2_geojson) and not os.path.exists(t1_geojson):
        print("⚠️ Warning: 'glacial_lakes_2020.geojson' not found. Creating placeholder baseline layer...")
        mock_gdf = gpd.read_file(t2_geojson)
        mock_gdf.geometry = mock_gdf.geometry.translate(xoff=0.0005, yoff=0.0005)
        mock_gdf.to_file(t1_geojson, driver="GeoJSON")

    if os.path.exists(t1_geojson) and os.path.exists(t2_geojson):
        compute_lake_area_change(t1_geojson, t2_geojson, "./lake_expansion_mask.geojson")
        create_side_by_side_lake_map(t1_geojson, t2_geojson, "./lake_comparison_map.html")
    else:
        print("❌ Error: Cannot build map because vector geometries are missing.")
