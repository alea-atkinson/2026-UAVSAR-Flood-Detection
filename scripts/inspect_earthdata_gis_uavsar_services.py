#!/usr/bin/env python3
"""Inspect NASA Earthdata GIS UAVSAR candidate item pages and services.

This script only probes public HTML, ArcGIS item JSON, and ArcGIS service/layer
metadata. It does not download raster/image data, create splits, or retrain.

Outputs:
  outputs/additional_data_inventory/earthdata_gis_service_inspection.md
  outputs/additional_data_inventory/earthdata_gis_service_urls.csv
  outputs/additional_data_inventory/earthdata_gis_layers.csv
  outputs/additional_data_inventory/earthdata_gis_raw_response_notes.txt
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "additional_data_inventory"

PORTAL = "https://gis.earthdata.nasa.gov/portal"
SHARING_REST = f"{PORTAL}/sharing/rest/content/items"

CANDIDATES = [
    {
        "candidate_id": "texas_2025_classified_vectors",
        "event": "Texas July 2025",
        "expected_title": "UAVSAR Classified Image Vectors for the Flooding in Texas July 2025",
        "item_id": "e58fef3c48494713945ec8845f7c2ae6",
        "item_page": f"{PORTAL}/home/item.html?id=e58fef3c48494713945ec8845f7c2ae6",
        "map_viewer": f"{PORTAL}/apps/mapviewer/index.html?layers=e58fef3c48494713945ec8845f7c2ae6",
    },
    {
        "candidate_id": "milton_2024_classification",
        "event": "Hurricane Milton 2024",
        "expected_title": "UAVSAR Imagery Classification after Hurricane Milton",
        "item_id": "a8ffa63288694e5dacfbbbfcd65187e0",
        "item_page": f"{PORTAL}/home/item.html?id=a8ffa63288694e5dacfbbbfcd65187e0",
        "map_viewer": "",
    },
    {
        "candidate_id": "milton_2024_rgb",
        "event": "Hurricane Milton 2024",
        "expected_title": "UAVSAR Imagery RGB after Hurricane Milton",
        "item_id": "912fb5f69b694d1a89914551a09bb27b",
        "item_page": f"{PORTAL}/home/item.html?id=912fb5f69b694d1a89914551a09bb27b",
        "map_viewer": "",
    },
    {
        "candidate_id": "dance_flood_extents_uavsar",
        "event": "DaNCE / Flood Extents from UAVSAR",
        "expected_title": "Flood Extents from UAVSAR",
        "item_id": "d1846fce068b48898b6a4d11361b862a",
        "item_page": f"{PORTAL}/home/item.html?id=d1846fce068b48898b6a4d11361b862a",
        "map_viewer": f"{PORTAL}/apps/mapviewer/index.html?layers=d1846fce068b48898b6a4d11361b862a",
    },
]

KEYWORDS = [
    "url",
    "serviceUrl",
    "MapServer",
    "FeatureServer",
    "ImageServer",
    "layers",
    "operationalLayers",
]


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def clean_text(value: Any, max_len: int = 500) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def csv_join(values: Any, max_len: int = 1200) -> str:
    if values is None or values == "":
        return ""
    if isinstance(values, (list, tuple, set)):
        text = "; ".join(clean_text(v, 180) for v in values)
    else:
        text = clean_text(values, max_len)
    return text[:max_len]


def build_url(url: str, **params: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query.update(params)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def request_url(url: str, *, expect_json: bool = False, referer: str = "") -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 uavsar-service-inspector/1.0",
        "Accept": "application/json,text/html,*/*",
    }
    if referer:
        headers["Referer"] = referer

    req = urllib.request.Request(url, headers=headers)
    result: dict[str, Any] = {
        "url": url,
        "ok": False,
        "status": "",
        "content_type": "",
        "text": "",
        "json": None,
        "error": "",
    }
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(1_500_000)
            charset = resp.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            result.update({
                "ok": 200 <= resp.status < 400,
                "status": str(resp.status),
                "content_type": resp.headers.get("Content-Type", ""),
                "text": text,
            })
    except urllib.error.HTTPError as exc:
        body = exc.read(8000).decode("utf-8", errors="replace")
        result.update({
            "status": str(exc.code),
            "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
            "text": body,
            "error": f"HTTPError: {exc}",
        })
    except (urllib.error.URLError, TimeoutError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if expect_json or "json" in result["content_type"].lower() or result["text"].lstrip().startswith(("{", "[")):
        try:
            result["json"] = json.loads(result["text"])
        except json.JSONDecodeError as exc:
            result["error"] = f"JSONDecodeError: {exc}"
    return result


def first_chars(result: dict[str, Any], n: int = 500) -> str:
    text = result.get("text", "") or ""
    return clean_text(text[:n], n)


def extract_service_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    decoded = html.unescape(text)
    decoded = decoded.replace("\\/", "/")
    decoded = urllib.parse.unquote(decoded)

    url_patterns = [
        r"https?://[^\"'<>\\\s]+?/(?:MapServer|FeatureServer|ImageServer)(?:/\d+)?",
        r"https?://[^\"'<>\\\s]+?/arcgis/rest/services/[^\"'<>\\\s]+?(?:MapServer|FeatureServer|ImageServer)(?:/\d+)?",
    ]
    found: list[str] = []
    for pattern in url_patterns:
        for match in re.finditer(pattern, decoded, flags=re.I):
            found.append(match.group(0).rstrip("),.;]}'\""))

    # ArcGIS pages sometimes store service URLs as JSON strings with escaped
    # entities that are easier to catch by looking for URL-ish values.
    for match in re.finditer(r'"(?:url|serviceUrl)"\s*:\s*"([^"]+)"', decoded, flags=re.I):
        value = match.group(1).replace("\\/", "/")
        if re.search(r"/(?:MapServer|FeatureServer|ImageServer)(?:/\d+)?", value, flags=re.I):
            found.append(value.rstrip("),.;]}'\""))

    return sorted(set(found))


def extract_keyword_snippets(text: str, expected_title: str) -> dict[str, str]:
    snippets: dict[str, str] = {}
    search_terms = KEYWORDS + [expected_title]
    for term in search_terms:
        lower_text = text.lower()
        lower_term = term.lower()
        idx = lower_text.find(lower_term)
        if idx == -1:
            snippets[term] = ""
            continue
        start = max(0, idx - 160)
        end = min(len(text), idx + len(term) + 260)
        snippets[term] = clean_text(text[start:end], 500)
    return snippets


def discover_urls_in_json(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"url", "serviceurl"} and isinstance(child, str):
                if re.search(r"/(?:MapServer|FeatureServer|ImageServer)(?:/\d+)?", child, flags=re.I):
                    found.append(child)
            found.extend(discover_urls_in_json(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(discover_urls_in_json(child))
    elif isinstance(value, str):
        found.extend(extract_service_urls_from_text(value))
    return sorted(set(found))


def service_type(service_url: str) -> str:
    match = re.search(r"/(MapServer|FeatureServer|ImageServer)(?:/\d+)?/?$", service_url, flags=re.I)
    return match.group(1) if match else "other"


def service_base_url(service_url: str) -> str:
    return re.sub(r"/\d+/?$", "", service_url.rstrip("/"))


def endpoint_probe(url: str, referer: str) -> str:
    result = request_url(build_url(url, f="json"), expect_json=True, referer=referer)
    if result["ok"] and not arcgis_error_message(result.get("json")):
        return "yes"
    if result["status"]:
        msg = arcgis_error_message(result.get("json")) or result.get("error") or first_chars(result, 180)
        return f"no/status {result['status']}: {clean_text(msg, 180)}"
    return f"no: {result.get('error', 'unknown error')}"


def arcgis_error_message(data: Any) -> str:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        details = csv_join(err.get("details"), 300)
        return clean_text(f"{err.get('code', '')} {err.get('message', '')} {details}", 300)
    return ""


def layer_candidates_from_service(service_json: dict[str, Any]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for key in ("layers", "tables"):
        value = service_json.get(key)
        if isinstance(value, list):
            layers.extend(item for item in value if isinstance(item, dict))
    return layers


def inspect_service(service_url: str, referer: str, candidate: dict[str, str]) -> tuple[dict[str, str], list[dict[str, str]], list[str]]:
    notes: list[str] = []
    base_url = service_base_url(service_url)
    svc_type = service_type(base_url)

    service_result = request_url(build_url(base_url, f="json"), expect_json=True, referer=referer)
    service_json = service_result.get("json") if isinstance(service_result.get("json"), dict) else {}
    service_error = arcgis_error_message(service_json) or service_result.get("error", "")

    if service_result["text"]:
        notes.append(f"SERVICE {base_url} status={service_result['status']} first500={first_chars(service_result)}")
    elif service_result["error"]:
        notes.append(f"SERVICE {base_url} error={service_result['error']}")

    layers = layer_candidates_from_service(service_json)
    query_available = "not checked"
    export_available = "not checked"

    if svc_type == "FeatureServer":
        if layers:
            query_available = endpoint_probe(f"{base_url}/{layers[0].get('id', 0)}/query", referer)
        else:
            query_available = endpoint_probe(f"{base_url}/0/query", referer)
        export_available = "feature query/rasterization path, not direct raster export"
    elif svc_type == "ImageServer":
        export_available = endpoint_probe(f"{base_url}/exportImage", referer)
        query_available = endpoint_probe(f"{base_url}/query", referer)
    elif svc_type == "MapServer":
        export_available = endpoint_probe(f"{base_url}/export", referer)
        query_available = endpoint_probe(f"{base_url}/0/query", referer) if layers else "not checked; no layers found"

    service_row = {
        "candidate_id": candidate["candidate_id"],
        "event": candidate["event"],
        "expected_title": candidate["expected_title"],
        "item_id": candidate["item_id"],
        "service_url": base_url,
        "service_type": svc_type,
        "service_status": service_result["status"],
        "service_error": service_error,
        "service_title": clean_text(service_json.get("name") or service_json.get("serviceDescription") or service_json.get("documentInfo", {}).get("Title", ""), 300),
        "capabilities": csv_join(service_json.get("capabilities")),
        "layer_count": str(len(layers)),
        "query_available": query_available,
        "export_available": export_available,
        "extent": json.dumps(service_json.get("fullExtent") or service_json.get("extent") or "", sort_keys=True)[:900],
    }

    layer_rows: list[dict[str, str]] = []
    for layer in layers:
        layer_id = str(layer.get("id", ""))
        layer_url = f"{base_url}/{layer_id}" if layer_id != "" else base_url
        layer_result = request_url(build_url(layer_url, f="json"), expect_json=True, referer=referer)
        layer_json = layer_result.get("json") if isinstance(layer_result.get("json"), dict) else {}
        layer_error = arcgis_error_message(layer_json) or layer_result.get("error", "")
        fields = []
        if isinstance(layer_json.get("fields"), list):
            fields = [f.get("name", "") for f in layer_json["fields"] if isinstance(f, dict)]
        layer_rows.append({
            "candidate_id": candidate["candidate_id"],
            "item_id": candidate["item_id"],
            "expected_title": candidate["expected_title"],
            "service_url": base_url,
            "service_type": svc_type,
            "layer_id": layer_id,
            "layer_name": clean_text(layer.get("name") or layer_json.get("name", ""), 300),
            "layer_type": clean_text(layer.get("type") or layer_json.get("type", ""), 200),
            "geometry_type": clean_text(layer_json.get("geometryType", ""), 160),
            "capabilities": csv_join(layer_json.get("capabilities")),
            "fields": csv_join(fields[:50], 1600),
            "extent": json.dumps(layer_json.get("extent") or layer.get("extent") or "", sort_keys=True)[:900],
            "layer_status": layer_result["status"],
            "layer_error": layer_error,
        })
        if layer_result["text"]:
            notes.append(f"LAYER {layer_url} status={layer_result['status']} first500={first_chars(layer_result)}")
    return service_row, layer_rows, notes


def infer_label_status(candidate: dict[str, str], service_rows: list[dict[str, str]], layer_rows: list[dict[str, str]], page_text: str) -> tuple[str, str, str]:
    text = " ".join([
        candidate["expected_title"],
        page_text[:5000],
        " ".join(r.get("service_title", "") for r in service_rows),
        " ".join(r.get("layer_name", "") + " " + r.get("fields", "") for r in layer_rows),
    ]).lower()

    has_uavsar = "uavsar" in text
    has_flood = any(term in text for term in ["flood", "inundation", "water", "classified", "classification", "extent"])
    is_rgb = "rgb" in text and not any(term in text for term in ["classified", "classification", "flood extent", "inundation"])
    has_vector = any(r.get("service_type") == "FeatureServer" or r.get("geometry_type") for r in layer_rows + service_rows)
    has_image = any(r.get("service_type") in {"ImageServer", "MapServer"} for r in service_rows)

    if not service_rows and has_uavsar and has_flood and not is_rgb:
        label_status = "candidate/page title suggests UAVSAR flood labels/classes; no service metadata found"
    elif not service_rows and is_rgb:
        label_status = "candidate/page title suggests imagery/RGB visualization, not labels; no service metadata found"
    elif has_uavsar and has_flood and not is_rgb:
        label_status = "looks like possible UAVSAR flood labels/classes, but verify class semantics"
    elif is_rgb:
        label_status = "looks like imagery/RGB visualization, not labels"
    else:
        label_status = "unknown from discovered metadata"

    if has_vector and has_flood:
        export_status = "likely rasterizable if query/export permissions and class fields are valid"
    elif has_image and has_flood:
        export_status = "possibly exportable image/map service; GeoTIFF export not confirmed"
    elif service_rows:
        export_status = "service found, but label export/rasterization path unclear"
    else:
        export_status = "no service URL found; manual browser inspection required"

    if not service_rows:
        next_action = "Open the item/map viewer in a browser dev console and inspect network requests for service URLs."
    elif "RGB" in candidate["expected_title"]:
        next_action = "Use only as companion imagery unless a separate classification/flood layer is discovered."
    elif "verify class semantics" in label_status:
        next_action = "Inspect layer fields/legend/classes, then test a small query/export window before any raster download."
    else:
        next_action = "Manually verify whether the service contains label classes or only visualization."
    return label_status, export_status, next_action


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None found._"
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [clean_text(row.get(key, ""), 180).replace("|", "\\|") for _, key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def inspect_candidate(candidate: dict[str, str]) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], list[str]]:
    notes: list[str] = [f"=== {candidate['candidate_id']} / {candidate['expected_title']} ==="]
    discovered_urls: set[str] = set()

    html_results = []
    for page_kind, url in [("item_page", candidate["item_page"]), ("map_viewer", candidate.get("map_viewer", ""))]:
        if not url:
            continue
        result = request_url(url, referer=PORTAL)
        html_results.append((page_kind, result))
        notes.append(f"{page_kind.upper()} {url} status={result['status']} error={result['error']}")
        notes.append(f"{page_kind.upper()} first500={first_chars(result)}")
        snippets = extract_keyword_snippets(result.get("text", ""), candidate["expected_title"])
        for key, snippet in snippets.items():
            if snippet:
                notes.append(f"{page_kind.upper()} snippet[{key}]={snippet}")
        discovered_urls.update(extract_service_urls_from_text(result.get("text", "")))

    rest_results = []
    rest_urls = [
        f"{SHARING_REST}/{candidate['item_id']}?f=json",
        f"{SHARING_REST}/{candidate['item_id']}/data?f=json",
    ]
    for url in rest_urls:
        result = request_url(url, expect_json=True, referer=candidate["item_page"])
        rest_results.append(result)
        notes.append(f"REST {url} status={result['status']} error={result['error']}")
        notes.append(f"REST first500={first_chars(result)}")
        if isinstance(result.get("json"), (dict, list)):
            discovered_urls.update(discover_urls_in_json(result["json"]))
        else:
            discovered_urls.update(extract_service_urls_from_text(result.get("text", "")))

    service_rows: list[dict[str, str]] = []
    layer_rows: list[dict[str, str]] = []
    for url in sorted(discovered_urls):
        service_row, layers, service_notes = inspect_service(url, candidate["item_page"], candidate)
        service_rows.append(service_row)
        layer_rows.extend(layers)
        notes.extend(service_notes)

    item_page_reachable = any(kind == "item_page" and result["ok"] for kind, result in html_results)
    rest_reachable = any(result["ok"] and not arcgis_error_message(result.get("json")) for result in rest_results)
    rest_errors = "; ".join(
        clean_text(arcgis_error_message(result.get("json")) or result.get("error") or first_chars(result, 160), 220)
        for result in rest_results
        if not (result["ok"] and not arcgis_error_message(result.get("json")))
    )
    page_text = " ".join(result.get("text", "")[:10000] for _kind, result in html_results)
    label_status, export_status, next_action = infer_label_status(candidate, service_rows, layer_rows, page_text)

    summary = {
        "candidate_id": candidate["candidate_id"],
        "event": candidate["event"],
        "expected_title": candidate["expected_title"],
        "item_id": candidate["item_id"],
        "item_page": candidate["item_page"],
        "map_viewer": candidate.get("map_viewer", ""),
        "item_page_reachable": "yes" if item_page_reachable else "no",
        "rest_item_metadata_reachable": "yes" if rest_reachable else "no",
        "rest_errors": rest_errors,
        "service_url_found": "yes" if service_rows else "no",
        "service_types": csv_join(sorted({r["service_type"] for r in service_rows})),
        "service_count": str(len(service_rows)),
        "layers_found": "yes" if layer_rows else "no",
        "layer_count": str(len(layer_rows)),
        "label_status": label_status,
        "export_or_rasterize_status": export_status,
        "recommended_next_action": next_action,
    }
    return summary, service_rows, layer_rows, notes


def write_markdown(path: Path, summaries: list[dict[str, str]], service_rows: list[dict[str, str]], layer_rows: list[dict[str, str]]) -> None:
    parts = [
        "# Earthdata GIS UAVSAR Service Inspection",
        "",
        f"Generated: `{now_utc()}`",
        "",
        "This is service/layer inspection only. No large rasters, training data, splits, or retraining were touched.",
        "",
        "## Candidate Answers",
        "",
    ]

    for summary in summaries:
        services = [r for r in service_rows if r["candidate_id"] == summary["candidate_id"]]
        layers = [r for r in layer_rows if r["candidate_id"] == summary["candidate_id"]]
        parts.extend([
            f"### {summary['expected_title']}",
            "",
            f"- Event: {summary['event']}",
            f"- Item page reachable: {summary['item_page_reachable']}",
            f"- REST item metadata reachable: {summary['rest_item_metadata_reachable']}",
            f"- REST errors/notes: {summary['rest_errors'] or 'none'}",
            f"- Service URL found: {summary['service_url_found']}",
            f"- Service type(s): {summary['service_types'] or 'none'}",
            f"- Layers found: {summary['layers_found']} ({summary['layer_count']})",
            f"- Label interpretation: {summary['label_status']}",
            f"- Export/rasterize interpretation: {summary['export_or_rasterize_status']}",
            f"- Recommended next action: {summary['recommended_next_action']}",
            "",
            "Services:",
            "",
            md_table(services, [
                ("Service URL", "service_url"),
                ("Type", "service_type"),
                ("Status", "service_status"),
                ("Layers", "layer_count"),
                ("Query", "query_available"),
                ("Export", "export_available"),
            ]),
            "",
            "Layers:",
            "",
            md_table(layers, [
                ("Layer ID", "layer_id"),
                ("Layer name", "layer_name"),
                ("Geometry", "geometry_type"),
                ("Capabilities", "capabilities"),
                ("Fields", "fields"),
            ]),
            "",
        ])

    parts.extend([
        "## Service URL Inventory",
        "",
        md_table(service_rows, [
            ("Candidate", "expected_title"),
            ("Service URL", "service_url"),
            ("Type", "service_type"),
            ("Status", "service_status"),
            ("Capabilities", "capabilities"),
            ("Error", "service_error"),
        ]),
        "",
        "## Caution",
        "",
        "A reachable item page does not guarantee a direct GeoTIFF download or reusable label mask. If no service URL was discovered here, manual browser inspection of network requests is still required.",
        "",
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, str]] = []
    service_rows: list[dict[str, str]] = []
    layer_rows: list[dict[str, str]] = []
    raw_notes: list[str] = [f"Generated: {now_utc()}"]

    for candidate in CANDIDATES:
        print(f"Inspecting {candidate['expected_title']}")
        summary, services, layers, notes = inspect_candidate(candidate)
        summaries.append(summary)
        service_rows.extend(services)
        layer_rows.extend(layers)
        raw_notes.extend(notes)
        raw_notes.append("")

    write_csv(
        OUT_DIR / "earthdata_gis_service_urls.csv",
        service_rows,
        [
            "candidate_id", "event", "expected_title", "item_id", "service_url", "service_type",
            "service_status", "service_error", "service_title", "capabilities", "layer_count",
            "query_available", "export_available", "extent",
        ],
    )
    write_csv(
        OUT_DIR / "earthdata_gis_layers.csv",
        layer_rows,
        [
            "candidate_id", "item_id", "expected_title", "service_url", "service_type",
            "layer_id", "layer_name", "layer_type", "geometry_type", "capabilities",
            "fields", "extent", "layer_status", "layer_error",
        ],
    )
    write_markdown(
        OUT_DIR / "earthdata_gis_service_inspection.md",
        summaries,
        service_rows,
        layer_rows,
    )
    (OUT_DIR / "earthdata_gis_raw_response_notes.txt").write_text("\n".join(raw_notes), encoding="utf-8")

    print(f"\nWrote {OUT_DIR / 'earthdata_gis_service_inspection.md'}")
    print(f"Wrote {OUT_DIR / 'earthdata_gis_service_urls.csv'}")
    print(f"Wrote {OUT_DIR / 'earthdata_gis_layers.csv'}")
    print(f"Wrote {OUT_DIR / 'earthdata_gis_raw_response_notes.txt'}")
    print("No large rasters, training data, splits, or retraining were touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
