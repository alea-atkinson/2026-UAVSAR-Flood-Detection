# Additional UAVSAR Flood Data Candidate Inventory

Audit generated: `2026-06-19T13:20:43+00:00`

This is a metadata/provenance audit only. It does not download large rasters, alter splits, or retrain models.

## Land Cover Pairing

U.S. event locations should be pairable with Esri 10m Land Use/Land Cover Time Series or USGS/MRLC NLCD. This audit records availability only; it does not download land cover.

## Ranked Candidate List

| Rank | Candidate | Event | Type | Label status | Export/download | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | UAVSAR Classified Image Vectors for the Flooding in Texas July 2025 | Texas Flooding, July 2025 |  | candidate title suggests UAVSAR + flood mask/classification; metadata not verified | not confirmed; ArcGIS item metadata was inaccessible or empty | low: UAVSAR suggested by title, flood labels suggested by title, rasterizable vectors |
| 3 | Flood Extents from UAVSAR | DaNCE March 2026 / Hurricane Florence UAVSAR products |  | candidate title suggests UAVSAR + flood mask/classification; metadata not verified | not confirmed; ArcGIS item metadata was inaccessible or empty | low: UAVSAR suggested by title, flood labels suggested by title, rasterizable vectors |
| 3 | Texas Flood Mapping Deployment, Aug 30-Sep 7 2017 | Hurricane Harvey / Texas Flood Mapping, Aug 30-Sep 7 2017 | web page / reference search | UAVSAR deployment/search page found; flood mask/classification product not confirmed | UAVSAR search/download page appears reachable; derived flood labels not confirmed | low: UAVSAR |
| 2 | UAVSAR Imagery Classification after Hurricane Milton | Hurricane Milton, October 2024 |  | candidate title suggests UAVSAR + flood mask/classification; metadata not verified | not confirmed; ArcGIS item metadata was inaccessible or empty | low: UAVSAR suggested by title, flood labels suggested by title |
| 1 | UAVSAR Classified Images for Hurricane Florence | Hurricane Florence UAVSAR products |  | candidate title suggests UAVSAR + flood mask/classification; metadata not verified | not confirmed; ArcGIS item metadata was inaccessible or empty | low: UAVSAR suggested by title, flood labels suggested by title, possible overlap |
| -1 | UAVSAR Flood Inundation Extent during Hurricane Florence | Hurricane Florence, 2018 | web page / reference search | candidate title suggests UAVSAR + flood mask/classification; metadata not verified | not confirmed from this audit; manual UAVSAR/Figshare follow-up needed | low: UAVSAR suggested by title, flood labels suggested by title, possible overlap, metadata fetch issue |
| -2 | UAVSAR Imagery RGB after Hurricane Milton | Hurricane Milton, October 2024 |  | candidate title suggests imagery/visualization only; metadata not verified | not confirmed; ArcGIS item metadata was inaccessible or empty | low: UAVSAR suggested by title, imagery only |

## Most Likely Usable Now

No candidate can be called ready without manual verification.

## UAVSAR + Flood Mask/Classification Candidates

No confirmed UAVSAR + flood-label candidate from metadata alone.

## Imagery or Visualization Only

| Candidate | Event | Reason | Use |
| --- | --- | --- | --- |
| UAVSAR Imagery RGB after Hurricane Milton | Hurricane Milton, October 2024 | candidate title suggests imagery/visualization only; metadata not verified | Likely UAVSAR RGB visualization/imagery; probably not labels by itself. |

## Likely Vector-to-Raster Conversion Needed

| Candidate | Event | Geometry | Layers | Conversion |
| --- | --- | --- | --- | --- |
| UAVSAR Classified Image Vectors for the Flooding in Texas July 2025 | Texas Flooding, July 2025 |  |  | possible vector-to-raster conversion; metadata not verified |
| Flood Extents from UAVSAR | DaNCE March 2026 / Hurricane Florence UAVSAR products |  |  | possible vector-to-raster conversion; metadata not verified |

## Worth Asking Dr. Jin About

| Candidate | Question | Notes/errors |
| --- | --- | --- |
| UAVSAR Classified Image Vectors for the Flooding in Texas July 2025 | not confirmed; ArcGIS item metadata was inaccessible or empty | item metadata error 400: Item does not exist or is inaccessible.; item data error 400: Item does not exist or is inaccessible. |
| Flood Extents from UAVSAR | not confirmed; ArcGIS item metadata was inaccessible or empty | item metadata error 400: Item does not exist or is inaccessible.; item data error 400: Item does not exist or is inaccessible. |
| Texas Flood Mapping Deployment, Aug 30-Sep 7 2017 | UAVSAR search/download page appears reachable; derived flood labels not confirmed |  |
| UAVSAR Imagery Classification after Hurricane Milton | not confirmed; ArcGIS item metadata was inaccessible or empty | item metadata error 400: Item does not exist or is inaccessible.; item data error 400: Item does not exist or is inaccessible. |
| UAVSAR Classified Images for Hurricane Florence | not confirmed; ArcGIS item metadata was inaccessible or empty | item metadata error 400: Item does not exist or is inaccessible.; item data error 400: Item does not exist or is inaccessible. |
| UAVSAR Flood Inundation Extent during Hurricane Florence | not confirmed from this audit; manual UAVSAR/Figshare follow-up needed | https://figshare.com/search?q=UAVSAR%20Flood%20Inundation%20Extent%20during%20Hurricane%20Florence: JSONDecodeError: Expecting value: line 1 column 1 (char 0); https://figshare.... |
| UAVSAR Imagery RGB after Hurricane Milton | not confirmed; ArcGIS item metadata was inaccessible or empty | item metadata error 400: Item does not exist or is inaccessible.; item data error 400: Item does not exist or is inaccessible. |

## Recommended Next Dataset To Try First

Start with **UAVSAR Classified Image Vectors for the Flooding in Texas July 2025**
(Texas Flooding, July 2025).

Reason: low: UAVSAR suggested by title, flood labels suggested by title, rasterizable vectors.

Caution: not confirmed; ArcGIS item metadata was inaccessible or empty.
If it is a web map/image service rather than a direct GeoTIFF download, treat it as a service
export/rasterization task and verify projection, pixel spacing, class semantics, nodata, and
license/provenance before adding it to any training split.

## ArcGIS Layer Inventory Snapshot

No ArcGIS layers were discovered or layer metadata was unavailable.

## Source URLs

- **UAVSAR Classified Image Vectors for the Flooding in Texas July 2025**: https://gis.earthdata.nasa.gov/portal/home/item.html?id=e58fef3c48494713945ec8845f7c2ae6
- **Flood Extents from UAVSAR**: https://gis.earthdata.nasa.gov/portal/home/item.html?id=d1846fce068b48898b6a4d11361b862a
- **Texas Flood Mapping Deployment, Aug 30-Sep 7 2017**: https://uavsar.jpl.nasa.gov/cgi-bin/data.pl?search=Harvey
- **UAVSAR Imagery Classification after Hurricane Milton**: https://gis.earthdata.nasa.gov/portal/home/item.html?id=a8ffa63288694e5dacfbbbfcd65187e0
- **UAVSAR Classified Images for Hurricane Florence**: https://gis.earthdata.nasa.gov/portal/home/item.html?id=c20811992eaa4e3cb0dd7c7c0bccd23
- **UAVSAR Flood Inundation Extent during Hurricane Florence**: https://figshare.com/search?q=UAVSAR%20Flood%20Inundation%20Extent%20during%20Hurricane%20Florence
- **UAVSAR Imagery RGB after Hurricane Milton**: https://gis.earthdata.nasa.gov/portal/home/item.html?id=912fb5f69b694d1a89914551a09bb27b
