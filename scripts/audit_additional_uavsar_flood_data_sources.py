#!/usr/bin/env python3
"""Audit possible additional UAVSAR flood data sources.

This is a metadata/provenance inventory only. It does not download imagery,
rasters, masks, land cover products, or modify any training splits.

Outputs:
  outputs/additional_data_inventory/uavsar_flood_data_candidates.csv
  outputs/additional_data_inventory/uavsar_flood_data_candidates.md
  outputs/additional_data_inventory/arcgis_item_layer_inventory.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "additional_data_inventory"

NASA_PORTAL = "https://gis.earthdata.nasa.gov/portal"
NASA_REST = f"{NASA_PORTAL}/sharing/rest/content/items"

ARCGIS_ITEMS = [
    {
        "candidate_id": "texas_2025_classified_vectors",
        "event": "Texas Flooding, July 2025",
        "expected_title": "UAVSAR Classified Image Vectors for the Flooding in Texas July 2025",
        "item_id": "e58fef3c48494713945ec8845f7c2ae6",
        "starting_assessment": "Likely UAVSAR-derived classified vector labels; check whether class attributes can be rasterized.",
    },
    {
        "candidate_id": "milton_2024_classification",
        "event": "Hurricane Milton, October 2024",
        "expected_title": "UAVSAR Imagery Classification after Hurricane Milton",
        "item_id": "a8ffa63288694e5dacfbbbfcd65187e0",
        "starting_assessment": "Likely UAVSAR-derived classification labels; check service/layers and export options.",
    },
    {
        "candidate_id": "milton_2024_rgb",
        "event": "Hurricane Milton, October 2024",
        "expected_title": "UAVSAR Imagery RGB after Hurricane Milton",
        "item_id": "912fb5f69b694d1a89914551a09bb27b",
        "starting_assessment": "Likely UAVSAR RGB visualization/imagery; probably not labels by itself.",
    },
    {
        "candidate_id": "dance_2026_flood_extents",
        "event": "DaNCE March 2026 / Hurricane Florence UAVSAR products",
        "expected_title": "Flood Extents from UAVSAR",
        "item_id": "d1846fce068b48898b6a4d11361b862a",
        "starting_assessment": "Likely flood-extent label product; check if vectors or raster/image service.",
    },
    {
        "candidate_id": "florence_classified_images",
        "event": "Hurricane Florence UAVSAR products",
        "expected_title": "UAVSAR Classified Images for Hurricane Florence",
        "item_id": "c20811992eaa4e3cb0dd7c7c0bccd23",
        "starting_assessment": "Likely overlaps current Florence/NC work; useful provenance and possible extra labels.",
    },
]

WEB_CANDIDATES = [
    {
        "candidate_id": "harvey_2017_uavsar_deployment",
        "event": "Hurricane Harvey / Texas Flood Mapping, Aug 30-Sep 7 2017",
        "source_kind": "uavsar_deployment",
        "expected_title": "Texas Flood Mapping Deployment, Aug 30-Sep 7 2017",
        "urls": [
            "https://uavsar.jpl.nasa.gov/cgi-bin/data.pl?search=Harvey",
            "https://uavsar.jpl.nasa.gov/cgi-bin/data.pl?search=Texas%20Flood%20Mapping",
            "https://uavsar.jpl.nasa.gov/cgi-bin/data.pl?search=flood",
        ],
        "starting_assessment": "Need to verify downloadable UAVSAR imagery and whether a derived flood product exists.",
    },
    {
        "candidate_id": "florence_figshare_flood_extent",
        "event": "Hurricane Florence, 2018",
        "source_kind": "figshare_reference",
        "expected_title": "UAVSAR Flood Inundation Extent during Hurricane Florence",
        "urls": [
            "https://figshare.com/search?q=UAVSAR%20Flood%20Inundation%20Extent%20during%20Hurricane%20Florence",
            "https://figshare.com/search?q=UAVSAR%20Hurricane%20Florence%20flood%20extent",
        ],
        "starting_assessment": "Reference/provenance source; may overlap the current Florence subset.",
    },
]

LAND_COVER_NOTE = (
    "U.S. event locations should be pairable with Esri 10m Land Use/Land Cover "
    "Time Series or USGS/MRLC NLCD. This audit records availability only; it "
    "does not download land cover."
)


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _clean_text(value: Any, max_len: int = 800) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def _epoch_ms_to_iso(value: Any) -> str:
    try:
        if value in (None, ""):
            return ""
        return dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.UTC).date().isoformat()
    except (TypeError, ValueError, OSError):
        return str(value)


def _csv_join(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return "; ".join(_clean_text(v, 120) for v in values)
    return _clean_text(values, 300)


def fetch_json(url: str, timeout: int = 30) -> tuple[dict[str, Any] | list[Any] | None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "uavsar-flood-inventory/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return json.loads(resp.read().decode(charset, errors="replace")), ""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def fetch_text(url: str, timeout: int = 30) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "uavsar-flood-inventory/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace"), ""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def rest_url(url: str, **params: str) -> str:
    parsed = urllib.parse.urlparse(url)
    existing = dict(urllib.parse.parse_qsl(parsed.query))
    existing.update(params)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(existing)))


def item_url(item_id: str) -> str:
    return f"{NASA_REST}/{item_id}?f=json"


def item_data_url(item_id: str) -> str:
    return f"{NASA_REST}/{item_id}/data?f=json"


def item_page_url(item_id: str) -> str:
    return f"{NASA_PORTAL}/home/item.html?id={item_id}"


def service_metadata_urls(service_url: str) -> list[str]:
    if not service_url:
        return []
    urls = [rest_url(service_url, f="json")]
    if re.search(r"/(FeatureServer|MapServer|ImageServer)(?:/\d+)?/?$", service_url, re.I):
        base = re.sub(r"/\d+/?$", "", service_url.rstrip("/"))
        urls.append(rest_url(f"{base}/layers", f="json"))
    return list(dict.fromkeys(urls))


def endpoint_available(service_url: str, endpoint: str) -> str:
    if not service_url:
        return "unknown"
    url = f"{service_url.rstrip('/')}/{endpoint.lstrip('/')}"
    data, err = fetch_json(rest_url(url, f="json"), timeout=20)
    if data is not None and not (isinstance(data, dict) and data.get("error")):
        return "yes"
    if err:
        return f"no ({err})"
    if isinstance(data, dict) and data.get("error"):
        return f"no ({data['error'].get('message', 'ArcGIS error')})"
    return "no"


def infer_content_flags(text: str, item_type: str, service_url: str) -> dict[str, str]:
    t = text.lower()
    flood_terms = ["flood", "inundation", "water extent", "flood extent", "classification", "classified"]
    uavsar_terms = ["uavsar", "synthetic aperture", "sar", "polarimetric"]
    rgb_terms = ["rgb", "visualization", "imagery"]
    vector_terms = ["vector", "feature", "polygon", "shapefile", "geojson", "featureserver"]
    raster_terms = ["image service", "imageserver", "raster", "geotiff", "tiff", "classified image", "mapserver"]

    contains_flood_mask = any(term in t for term in flood_terms)
    contains_uavsar = any(term in t for term in uavsar_terms)
    imagery_only = any(term in t for term in rgb_terms) and not contains_flood_mask
    likely_vector = any(term in t for term in vector_terms) or "Feature" in item_type
    likely_raster = any(term in t for term in raster_terms) or "ImageServer" in service_url or "MapServer" in service_url

    if contains_flood_mask and contains_uavsar:
        label_status = "likely UAVSAR + flood mask/classification"
    elif contains_flood_mask:
        label_status = "likely flood mask/classification; UAVSAR linkage needs verification"
    elif imagery_only or "rgb" in t:
        label_status = "likely imagery/visualization only, not labels"
    else:
        label_status = "unclear from metadata"

    conversion = "unknown"
    if likely_vector and contains_flood_mask:
        conversion = "likely vector-to-raster conversion needed"
    elif likely_raster and contains_flood_mask:
        conversion = "possibly raster/image service; GeoTIFF export must be verified"
    elif imagery_only:
        conversion = "not label data"

    return {
        "contains_uavsar": "yes" if contains_uavsar else "unclear",
        "contains_flood_mask_or_classes": "yes" if contains_flood_mask else "unclear",
        "appears_imagery_only": "yes" if imagery_only else "no",
        "label_status": label_status,
        "conversion_need": conversion,
    }


def soften_unverified_flags(flags: dict[str, str], verified_metadata: bool) -> dict[str, str]:
    """Avoid presenting user-supplied candidate names as confirmed metadata."""
    if verified_metadata:
        return flags
    softened = dict(flags)
    if softened["contains_uavsar"] == "yes":
        softened["contains_uavsar"] = "suggested_by_title"
    if softened["contains_flood_mask_or_classes"] == "yes":
        softened["contains_flood_mask_or_classes"] = "suggested_by_title"
    if softened["appears_imagery_only"] == "yes":
        softened["appears_imagery_only"] = "suggested_by_title"
    if "likely UAVSAR + flood mask/classification" in softened["label_status"]:
        softened["label_status"] = "candidate title suggests UAVSAR + flood mask/classification; metadata not verified"
    elif "likely flood mask/classification" in softened["label_status"]:
        softened["label_status"] = "candidate title suggests flood mask/classification; metadata not verified"
    elif "likely imagery/visualization only" in softened["label_status"]:
        softened["label_status"] = "candidate title suggests imagery/visualization only; metadata not verified"
    if "vector-to-raster" in softened["conversion_need"]:
        softened["conversion_need"] = "possible vector-to-raster conversion; metadata not verified"
    elif softened["conversion_need"] != "not label data":
        softened["conversion_need"] = "unknown; metadata not verified"
    return softened


def score_candidate(row: dict[str, str]) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    text = " ".join(row.get(k, "") for k in row)
    lower = text.lower()

    if row.get("contains_uavsar") == "yes":
        score += 3
        reasons.append("UAVSAR")
    elif row.get("contains_uavsar") == "suggested_by_title":
        score += 1
        reasons.append("UAVSAR suggested by title")
    if row.get("contains_flood_mask_or_classes") == "yes":
        score += 4
        reasons.append("flood labels")
    elif row.get("contains_flood_mask_or_classes") == "suggested_by_title":
        score += 1
        reasons.append("flood labels suggested by title")
    if "vector-to-raster" in row.get("conversion_need", ""):
        score += 1
        reasons.append("rasterizable vectors")
    if "export" in lower and "yes" in lower:
        score += 1
        reasons.append("export/query signs")
    if row.get("appears_imagery_only") in {"yes", "suggested_by_title"}:
        score -= 3
        reasons.append("imagery only")
    if "florence" in lower and "overlap" in lower:
        score -= 1
        reasons.append("possible overlap")
    if "error:" in lower or "urlerror" in lower:
        score -= 2
        reasons.append("metadata fetch issue")

    if score >= 7:
        rank = "high"
    elif score >= 4:
        rank = "medium"
    else:
        rank = "low"
    return score, f"{rank}: {', '.join(reasons) if reasons else 'needs manual verification'}"


def summarize_layers(service_url: str, service_meta: dict[str, Any], layers_meta: dict[str, Any] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    layer_sources: list[dict[str, Any]] = []
    if isinstance(service_meta.get("layers"), list):
        layer_sources.extend(service_meta["layers"])
    if layers_meta and isinstance(layers_meta.get("layers"), list):
        layer_sources.extend(layers_meta["layers"])

    seen: set[tuple[str, str]] = set()
    for layer in layer_sources:
        layer_id = str(layer.get("id", ""))
        layer_name = _clean_text(layer.get("name", ""), 220)
        key = (layer_id, layer_name)
        if key in seen:
            continue
        seen.add(key)

        layer_detail = {}
        if service_url and layer_id != "":
            detail_url = rest_url(f"{service_url.rstrip('/')}/{layer_id}", f="json")
            detail, _err = fetch_json(detail_url, timeout=20)
            if isinstance(detail, dict):
                layer_detail = detail

        rows.append({
            "layer_id": layer_id,
            "layer_name": layer_name,
            "layer_type": _clean_text(layer.get("type") or layer_detail.get("type", ""), 160),
            "geometry_type": _clean_text(layer_detail.get("geometryType", ""), 160),
            "capabilities": _csv_join(layer_detail.get("capabilities") or service_meta.get("capabilities")),
            "fields": _csv_join([f.get("name") for f in layer_detail.get("fields", [])[:25] if isinstance(f, dict)]),
            "extent": json.dumps(layer_detail.get("extent") or layer.get("extent") or "", sort_keys=True)[:700],
        })
    return rows


def audit_arcgis_item(candidate: dict[str, str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    item_id = candidate["item_id"]
    metadata, item_err = fetch_json(item_url(item_id))
    item_data, data_err = fetch_json(item_data_url(item_id))

    metadata_errors: list[str] = []
    if isinstance(metadata, dict) and metadata.get("error"):
        err = metadata["error"]
        metadata_errors.append(
            f"item metadata error {err.get('code', '')}: {err.get('message', 'ArcGIS item error')}"
        )
        if err.get("details"):
            metadata_errors.append(_csv_join(err.get("details")))
        metadata = {}
    if isinstance(item_data, dict) and item_data.get("error"):
        err = item_data["error"]
        metadata_errors.append(
            f"item data error {err.get('code', '')}: {err.get('message', 'ArcGIS item data error')}"
        )
        if err.get("details"):
            metadata_errors.append(_csv_join(err.get("details")))
        item_data = {}

    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(item_data, dict):
        item_data = {}

    service_url = _clean_text(metadata.get("url", ""), 500)
    service_type = ""
    if "FeatureServer" in service_url:
        service_type = "FeatureServer"
    elif "ImageServer" in service_url:
        service_type = "ImageServer"
    elif "MapServer" in service_url:
        service_type = "MapServer"

    service_meta: dict[str, Any] = {}
    layers_meta: dict[str, Any] | None = None
    service_errors: list[str] = []
    for url in service_metadata_urls(service_url):
        data, err = fetch_json(url)
        if isinstance(data, dict):
            if url.endswith("layers?f=json") or "/layers?" in url:
                layers_meta = data
            else:
                service_meta = data
        elif err:
            service_errors.append(f"{url}: {err}")

    layer_rows = summarize_layers(service_url, service_meta, layers_meta)
    layer_names = "; ".join(f"{r['layer_id']}:{r['layer_name']}" for r in layer_rows)
    layer_count = str(len(layer_rows) or metadata.get("numViews", ""))

    text_blob = " ".join([
        candidate.get("expected_title", ""),
        candidate.get("event", ""),
        candidate.get("starting_assessment", ""),
        _clean_text(metadata.get("title", ""), 500),
        _clean_text(metadata.get("snippet", ""), 1000),
        _clean_text(metadata.get("description", ""), 2500),
        _clean_text(metadata.get("tags", ""), 1000),
        metadata.get("type", ""),
        service_url,
        json.dumps(service_meta)[:3000] if service_meta else "",
        json.dumps(item_data)[:3000] if item_data else "",
    ])
    verified_metadata = bool(metadata.get("id") or metadata.get("title") or service_url)
    flags = soften_unverified_flags(
        infer_content_flags(text_blob, metadata.get("type", ""), service_url),
        verified_metadata,
    )

    export_available = "unknown"
    query_available = "unknown"
    if service_type == "FeatureServer":
        query_available = endpoint_available(f"{service_url.rstrip('/')}/0", "query")
        export_available = "features can usually be queried; rasterization/export would be separate"
    elif service_type in {"ImageServer", "MapServer"}:
        export_available = endpoint_available(service_url, "exportImage" if service_type == "ImageServer" else "export")
        query_available = endpoint_available(service_url, "query")

    row = {
        "candidate_id": candidate["candidate_id"],
        "source_kind": "arcgis_item",
        "event": candidate["event"],
        "expected_title": candidate["expected_title"],
        "item_id": item_id,
        "item_page_url": item_page_url(item_id),
        "metadata_url": item_url(item_id),
        "title": _clean_text(metadata.get("title") or candidate["expected_title"], 300),
        "item_type": _clean_text(metadata.get("type", ""), 200),
        "owner": _clean_text(metadata.get("owner", ""), 160),
        "created": _epoch_ms_to_iso(metadata.get("created")),
        "modified": _epoch_ms_to_iso(metadata.get("modified")),
        "summary": _clean_text(metadata.get("snippet", ""), 900),
        "description": _clean_text(metadata.get("description", ""), 1200),
        "tags": _csv_join(metadata.get("tags")),
        "service_url": service_url,
        "service_type": service_type or _clean_text(service_meta.get("type", ""), 120),
        "layer_count": layer_count,
        "layer_names": layer_names,
        "geometry_type": _csv_join({r["geometry_type"] for r in layer_rows if r["geometry_type"]}),
        "capabilities": _csv_join(service_meta.get("capabilities")),
        "query_available": query_available,
        "export_available": export_available,
        "download_or_export_possible": infer_download_status(metadata, service_url, service_type, export_available, query_available, verified_metadata),
        "contains_uavsar": flags["contains_uavsar"],
        "contains_flood_mask_or_classes": flags["contains_flood_mask_or_classes"],
        "appears_imagery_only": flags["appears_imagery_only"],
        "label_status": flags["label_status"],
        "conversion_need": flags["conversion_need"],
        "land_cover_pairing": LAND_COVER_NOTE,
        "notes": candidate["starting_assessment"],
        "metadata_errors": "; ".join(e for e in [item_err, data_err, *metadata_errors, *service_errors] if e),
        "audit_timestamp_utc": _now_utc(),
    }
    score, rank_reason = score_candidate(row)
    row["rank_score"] = str(score)
    row["rank_reason"] = rank_reason

    for layer_row in layer_rows:
        layer_row.update({
            "candidate_id": candidate["candidate_id"],
            "item_id": item_id,
            "item_title": row["title"],
            "service_url": service_url,
        })

    return row, layer_rows


def infer_download_status(metadata: dict[str, Any], service_url: str, service_type: str, export_available: str, query_available: str, verified_metadata: bool = True) -> str:
    if not verified_metadata:
        return "not confirmed; ArcGIS item metadata was inaccessible or empty"
    item_type = metadata.get("type", "")
    access_info = " ".join([
        item_type,
        service_url,
        _clean_text(metadata.get("url", ""), 300),
        _clean_text(metadata.get("accessInformation", ""), 300),
    ]).lower()
    if "download" in access_info or "geotiff" in access_info or "tif" in access_info:
        return "possible, but verify file size/licensing before download"
    if service_type == "FeatureServer" and query_available.startswith("yes"):
        return "query possible; labels likely need vector export/rasterization rather than direct GeoTIFF"
    if service_type in {"ImageServer", "MapServer"} and export_available.startswith("yes"):
        return "service export endpoint appears available; direct GeoTIFF download not confirmed"
    if service_url:
        return "service URL exists, but export/download not confirmed"
    return "not confirmed from item metadata"


def audit_web_candidate(candidate: dict[str, Any]) -> dict[str, str]:
    notes: list[str] = [candidate["starting_assessment"]]
    titles: list[str] = []
    found_urls: list[str] = []
    errors: list[str] = []

    if candidate["source_kind"] == "figshare_reference":
        for url in candidate["urls"]:
            data, err = fetch_json(url)
            if err:
                text, text_err = fetch_text(url)
                if text:
                    found_urls.append(url)
                    titles.append(candidate["expected_title"])
                    notes.append(f"Figshare page/API URL probed but exact API search result not confirmed: {url}")
                    continue
                errors.append(f"{url}: {err or text_err}")
                continue
            if isinstance(data, list):
                for article in data[:5]:
                    if not isinstance(article, dict):
                        continue
                    titles.append(_clean_text(article.get("title", ""), 300))
                    doi = article.get("doi") or ""
                    article_url = article.get("url_public_html") or article.get("url") or ""
                    if article_url:
                        found_urls.append(article_url)
                    notes.append(
                        "Figshare search hit: "
                        + "; ".join(part for part in [
                            _clean_text(article.get("title", ""), 300),
                            f"doi={doi}" if doi else "",
                            f"published={article.get('published_date', '')}" if article.get("published_date") else "",
                        ] if part)
                    )
    else:
        for url in candidate["urls"]:
            text, err = fetch_text(url)
            if err:
                errors.append(f"{url}: {err}")
                continue
            found_urls.append(url)
            title_match = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
            if title_match:
                titles.append(_clean_text(title_match.group(1), 300))
            lower = text.lower()
            if "download" in lower or "search" in lower:
                notes.append(f"Page probed: {url}; contains search/download wording.")
            if "flood" in lower or "harvey" in lower:
                notes.append(f"Page probed: {url}; contains event/flood wording.")

    text_blob = " ".join([candidate["expected_title"], candidate["event"], " ".join(titles), " ".join(notes)])
    verified_metadata = bool(titles and found_urls and not errors)
    flags = soften_unverified_flags(infer_content_flags(text_blob, "", ""), verified_metadata)
    if candidate["source_kind"] == "uavsar_deployment":
        flags["contains_uavsar"] = "yes" if found_urls else "suggested_by_title"
        flags["contains_flood_mask_or_classes"] = "unclear"
        flags["appears_imagery_only"] = "no"
        flags["label_status"] = "UAVSAR deployment/search page found; flood mask/classification product not confirmed"
        flags["conversion_need"] = "unknown until product list/download metadata are reviewed"
    row = {
        "candidate_id": candidate["candidate_id"],
        "source_kind": candidate["source_kind"],
        "event": candidate["event"],
        "expected_title": candidate["expected_title"],
        "item_id": "",
        "item_page_url": found_urls[0] if found_urls else candidate["urls"][0],
        "metadata_url": "; ".join(candidate["urls"]),
        "title": titles[0] if titles else candidate["expected_title"],
        "item_type": "web page / reference search",
        "owner": "",
        "created": "",
        "modified": "",
        "summary": "",
        "description": " ".join(notes)[:1200],
        "tags": "",
        "service_url": "",
        "service_type": "",
        "layer_count": "",
        "layer_names": "",
        "geometry_type": "",
        "capabilities": "",
        "query_available": "not applicable",
        "export_available": "not confirmed",
        "download_or_export_possible": (
            "UAVSAR search/download page appears reachable; derived flood labels not confirmed"
            if candidate["source_kind"] == "uavsar_deployment" and found_urls
            else "not confirmed from this audit; manual UAVSAR/Figshare follow-up needed"
        ),
        "contains_uavsar": flags["contains_uavsar"],
        "contains_flood_mask_or_classes": flags["contains_flood_mask_or_classes"],
        "appears_imagery_only": flags["appears_imagery_only"],
        "label_status": flags["label_status"],
        "conversion_need": flags["conversion_need"],
        "land_cover_pairing": LAND_COVER_NOTE,
        "notes": " ".join(notes),
        "metadata_errors": "; ".join(errors),
        "audit_timestamp_utc": _now_utc(),
    }
    score, rank_reason = score_candidate(row)
    row["rank_score"] = str(score)
    row["rank_reason"] = rank_reason
    return row


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    lines = []
    headers = [name for name, _key in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        vals = []
        for _name, key in columns:
            value = _clean_text(row.get(key, ""), 180).replace("|", "\\|")
            vals.append(value)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, str]], layer_rows: list[dict[str, str]]) -> None:
    ranked = sorted(rows, key=lambda r: int(r.get("rank_score", "0") or 0), reverse=True)
    likely_now = [
        r for r in ranked
        if r.get("contains_uavsar") == "yes"
        and r.get("contains_flood_mask_or_classes") == "yes"
        and r.get("appears_imagery_only") != "yes"
    ]
    imagery_only = [r for r in ranked if r.get("appears_imagery_only") == "yes" or "RGB" in r.get("expected_title", "")]
    vector_needed = [r for r in ranked if "vector-to-raster" in r.get("conversion_need", "")]
    ask_dr_jin = [
        r for r in ranked
        if r.get("metadata_errors")
        or "not confirmed" in r.get("download_or_export_possible", "")
        or "overlap" in r.get("notes", "").lower()
        or r.get("source_kind") in {"uavsar_deployment", "figshare_reference"}
    ]

    next_dataset = likely_now[0] if likely_now else ranked[0]

    content = [
        "# Additional UAVSAR Flood Data Candidate Inventory",
        "",
        f"Audit generated: `{_now_utc()}`",
        "",
        "This is a metadata/provenance audit only. It does not download large rasters, alter splits, or retrain models.",
        "",
        "## Land Cover Pairing",
        "",
        LAND_COVER_NOTE,
        "",
        "## Ranked Candidate List",
        "",
        md_table(ranked, [
            ("Rank", "rank_score"),
            ("Candidate", "expected_title"),
            ("Event", "event"),
            ("Type", "item_type"),
            ("Label status", "label_status"),
            ("Export/download", "download_or_export_possible"),
            ("Why", "rank_reason"),
        ]),
        "",
        "## Most Likely Usable Now",
        "",
        md_table(likely_now, [
            ("Candidate", "expected_title"),
            ("Service type", "service_type"),
            ("Layers", "layer_names"),
            ("Conversion", "conversion_need"),
            ("Caution", "download_or_export_possible"),
        ]) if likely_now else "No candidate can be called ready without manual verification.",
        "",
        "## UAVSAR + Flood Mask/Classification Candidates",
        "",
        md_table(likely_now, [
            ("Candidate", "expected_title"),
            ("Event", "event"),
            ("Mask/classes", "contains_flood_mask_or_classes"),
            ("UAVSAR", "contains_uavsar"),
            ("Notes", "notes"),
        ]) if likely_now else "No confirmed UAVSAR + flood-label candidate from metadata alone.",
        "",
        "## Imagery or Visualization Only",
        "",
        md_table(imagery_only, [
            ("Candidate", "expected_title"),
            ("Event", "event"),
            ("Reason", "label_status"),
            ("Use", "notes"),
        ]) if imagery_only else "No imagery-only candidate was clearly identified.",
        "",
        "## Likely Vector-to-Raster Conversion Needed",
        "",
        md_table(vector_needed, [
            ("Candidate", "expected_title"),
            ("Event", "event"),
            ("Geometry", "geometry_type"),
            ("Layers", "layer_names"),
            ("Conversion", "conversion_need"),
        ]) if vector_needed else "No candidate clearly required vector-to-raster conversion from metadata alone.",
        "",
        "## Worth Asking Dr. Jin About",
        "",
        md_table(ask_dr_jin, [
            ("Candidate", "expected_title"),
            ("Question", "download_or_export_possible"),
            ("Notes/errors", "metadata_errors"),
        ]) if ask_dr_jin else "No special follow-up items identified.",
        "",
        "## Recommended Next Dataset To Try First",
        "",
        textwrap.dedent(f"""
        Start with **{next_dataset.get('expected_title', next_dataset.get('title', 'the highest-ranked candidate'))}**
        ({next_dataset.get('event', '')}).

        Reason: {next_dataset.get('rank_reason', 'highest-ranked by metadata heuristics')}.

        Caution: {next_dataset.get('download_or_export_possible', 'download/export not confirmed')}.
        If it is a web map/image service rather than a direct GeoTIFF download, treat it as a service
        export/rasterization task and verify projection, pixel spacing, class semantics, nodata, and
        license/provenance before adding it to any training split.
        """).strip(),
        "",
        "## ArcGIS Layer Inventory Snapshot",
        "",
        md_table(layer_rows, [
            ("Item", "item_title"),
            ("Layer ID", "layer_id"),
            ("Layer name", "layer_name"),
            ("Geometry", "geometry_type"),
            ("Capabilities", "capabilities"),
        ]) if layer_rows else "No ArcGIS layers were discovered or layer metadata was unavailable.",
        "",
        "## Source URLs",
        "",
    ]

    for row in ranked:
        content.append(f"- **{row['expected_title']}**: {row.get('item_page_url') or row.get('metadata_url')}")
    content.append("")

    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, str]] = []
    layer_rows: list[dict[str, str]] = []

    print("Auditing ArcGIS/NASA Earthdata GIS items...")
    for candidate in ARCGIS_ITEMS:
        print(f"  - {candidate['expected_title']}")
        row, layers = audit_arcgis_item(candidate)
        candidate_rows.append(row)
        layer_rows.extend(layers)

    print("Auditing UAVSAR/Figshare reference pages...")
    for candidate in WEB_CANDIDATES:
        print(f"  - {candidate['expected_title']}")
        candidate_rows.append(audit_web_candidate(candidate))

    candidate_fields = [
        "candidate_id", "rank_score", "rank_reason", "source_kind", "event", "expected_title",
        "item_id", "item_page_url", "metadata_url", "title", "item_type", "owner", "created",
        "modified", "summary", "description", "tags", "service_url", "service_type",
        "layer_count", "layer_names", "geometry_type", "capabilities", "query_available",
        "export_available", "download_or_export_possible", "contains_uavsar",
        "contains_flood_mask_or_classes", "appears_imagery_only", "label_status",
        "conversion_need", "land_cover_pairing", "notes", "metadata_errors",
        "audit_timestamp_utc",
    ]
    layer_fields = [
        "candidate_id", "item_id", "item_title", "service_url", "layer_id", "layer_name",
        "layer_type", "geometry_type", "capabilities", "fields", "extent",
    ]

    candidates_csv = OUT_DIR / "uavsar_flood_data_candidates.csv"
    layers_csv = OUT_DIR / "arcgis_item_layer_inventory.csv"
    markdown_path = OUT_DIR / "uavsar_flood_data_candidates.md"

    write_csv(candidates_csv, candidate_rows, candidate_fields)
    write_csv(layers_csv, layer_rows, layer_fields)
    write_markdown(markdown_path, candidate_rows, layer_rows)

    print(f"\nWrote {candidates_csv}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {layers_csv}")
    print("\nNo rasters, masks, land cover, or training splits were downloaded or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
