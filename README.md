# Glacial Frontier: Automated Real-Time Outburst Flood Vectorization Mapping Pipeline

An automated, remote sensing data engineering workflow designed to connect securely to NASA Earthdata repositories, ingest high-resolution Harmonized Landsat Sentinel-2 (HLS) multi-spectral streams, compute Normalized Difference Water Index (NDWI) pixel matrices, vectorize fragile lake geometry polygons, and generate interactive dual-timeline split-screen dashboards for disaster response mitigation.

## 📁 Repository Structure
- `pipeline.py`: The main automated satellite ingestion, array computation, and mapping execution core.
- `lake_comparison_map.html`: The compiled interactive dual-timeline split-screen map canvas.

## 🛠️ The Technical Stack
- **Data Ingestion Engine:** `earthaccess` (NASA Earthdata Login automated token management)
- **Raster Processing Core:** `rasterio`, `numpy` (NDWI Spectral Index execution matrices)
- **Geospatial Vectorization Architecture:** `geopandas`, `shapely` (Polygon conversion metrics)
- **Interactive UI Front-End Canvas:** `folium`, `SideBySideLayers` (Dual-view evaluation dashboard)

## 📊 Live Run Analytics Core Benchmarks
- **Total Surface Features Extracted:** 38,051 distinct alpine water bodies vectorized dynamically.
- **Computed Core Region Mass Base:** 4,030.452 square kilometers of active surface area evaluated.
- **Timeline Delta Core Output:** Automated UTM metric coordinate change evaluation framework.
