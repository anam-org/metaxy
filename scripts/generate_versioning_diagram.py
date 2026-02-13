#!/usr/bin/env python3
"""Generate docs/assets/diagrams/versioning.excalidraw from code.

Loads the feature graph from examples/example-overview/ and builds
diagram content from real feature/field names, dependencies, and versions.

Usage:
    uv run python scripts/generate_versioning_diagram.py
"""

from __future__ import annotations

import json
import uuid
import warnings
from pathlib import Path
from typing import Any

import metaxy as mx
from metaxy import FeatureGraph, FeatureKey, FieldKey
from metaxy.models.plan import FQFieldKey

# -- Colors --
BLUE = "#a5d8ff"
YELLOW = "#fff3bf"
GREEN = "#b2f2bb"
WHITE = "#ffffff"

# -- Layout constants --
ZONE_GAP = 120
ZONE_WIDTH = 1000
ZONE_HEIGHT = 750
TOP_LEFT = (50, 80)


def _id() -> str:
    return uuid.uuid4().hex[:20]


def _make_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    bg: str = WHITE,
    font_size: int = 20,
) -> list[dict[str, Any]]:
    """Return a rectangle element with a bound text label."""
    rect_id = _id()
    text_id = _id()
    rect: dict[str, Any] = {
        "id": rect_id,
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "strokeColor": "#1e1e1e",
        "backgroundColor": bg,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "roughness": 1,
        "opacity": 100,
        "angle": 0,
        "strokeStyle": "solid",
        "roundness": {"type": 3},
        "seed": hash(rect_id) % (2**31),
        "version": 1,
        "isDeleted": False,
        "boundElements": [{"id": text_id, "type": "text"}],
        "link": None,
        "locked": False,
        "groupIds": [],
        "frameId": None,
        "index": "a0",
        "updated": 1,
    }
    text: dict[str, Any] = {
        "id": text_id,
        "type": "text",
        "x": x + 10,
        "y": y + h / 2 - font_size * 0.7,
        "width": w - 20,
        "height": font_size * 1.4,
        "text": label,
        "fontSize": font_size,
        "fontFamily": 5,
        "textAlign": "center",
        "verticalAlign": "middle",
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "roughness": 1,
        "opacity": 100,
        "angle": 0,
        "strokeStyle": "solid",
        "roundness": None,
        "seed": hash(text_id) % (2**31),
        "version": 1,
        "isDeleted": False,
        "boundElements": None,
        "link": None,
        "locked": False,
        "groupIds": [],
        "frameId": None,
        "containerId": rect_id,
        "originalText": label,
        "autoResize": True,
        "lineHeight": 1.25,
        "index": "a0",
        "updated": 1,
    }
    return [rect, text]


def _make_title(x: float, y: float, label: str) -> dict[str, Any]:
    """Return a large bold zone title text element."""
    text_id = _id()
    return {
        "id": text_id,
        "type": "text",
        "x": x,
        "y": y,
        "width": 400,
        "height": 40,
        "text": label,
        "fontSize": 28,
        "fontFamily": 5,
        "textAlign": "left",
        "verticalAlign": "top",
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "roughness": 1,
        "opacity": 100,
        "angle": 0,
        "strokeStyle": "solid",
        "roundness": None,
        "seed": hash(text_id) % (2**31),
        "version": 1,
        "isDeleted": False,
        "boundElements": None,
        "link": None,
        "locked": False,
        "groupIds": [],
        "frameId": None,
        "index": "a0",
        "updated": 1,
    }


def _make_arrow(
    start_id: str,
    end_id: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> dict[str, Any]:
    """Return an arrow element connecting two shapes."""
    arrow_id = _id()
    return {
        "id": arrow_id,
        "type": "arrow",
        "x": start_x,
        "y": start_y,
        "width": abs(end_x - start_x),
        "height": abs(end_y - start_y),
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "roughness": 1,
        "opacity": 100,
        "angle": 0,
        "strokeStyle": "solid",
        "roundness": {"type": 2},
        "seed": hash(arrow_id) % (2**31),
        "version": 1,
        "isDeleted": False,
        "boundElements": None,
        "link": None,
        "locked": False,
        "groupIds": [],
        "frameId": None,
        "startBinding": {
            "elementId": start_id,
            "focus": 0,
            "gap": 4,
            "fixedPoint": None,
        },
        "endBinding": {
            "elementId": end_id,
            "focus": 0,
            "gap": 4,
            "fixedPoint": None,
        },
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "points": [[0, 0], [end_x - start_x, end_y - start_y]],
        "elbowed": False,
        "index": "a0",
        "updated": 1,
    }


def _rect_id(elements: list[dict[str, Any]]) -> str:
    """Get the rectangle element ID (first element) from a _make_rect result."""
    return elements[0]["id"]


def _rect_bottom_center(x: float, y: float, w: float, h: float) -> tuple[float, float]:
    return (x + w / 2, y + h)


def _rect_top_center(x: float, y: float, w: float, _h: float) -> tuple[float, float]:
    return (x + w / 2, y)


# -- Zone builders --


def _build_zone1(ox: float, oy: float, graph: FeatureGraph) -> list[dict[str, Any]]:
    """Zone 1: Field Version — concrete example: Crop.audio."""
    crop_key = FeatureKey("example/crop")
    video_key = FeatureKey("example/video")

    crop_defn = graph.get_feature_definition(crop_key)
    audio_field = next(f for f in crop_defn.spec.fields if f.key.to_string() == "audio")

    video_audio_ver = graph.get_field_version(FQFieldKey(feature=video_key, field=FieldKey("audio")))
    crop_audio_ver = graph.get_field_version(FQFieldKey(feature=crop_key, field=FieldKey("audio")))

    crop_fq = crop_key.to_string()
    video_fq = video_key.to_string()

    elements: list[dict[str, Any]] = []
    elements.append(_make_title(ox, oy, "Field Version"))

    row_y = oy + 60
    bw, bh = 200, 55
    gap = 30

    fq_x = ox + 20
    cv_x = fq_x + bw + gap
    pv_w = bw + 40
    pv_x = cv_x + bw + gap

    fq = _make_rect(fq_x, row_y, bw, bh, label=f"{crop_fq}:audio\n(FQ field key)", bg=BLUE, font_size=16)
    cv = _make_rect(cv_x, row_y, bw, bh, label=f'code_version\n"{audio_field.code_version}"', bg=BLUE, font_size=16)
    pv = _make_rect(
        pv_x, row_y, pv_w, bh, label=f'{video_fq}:audio version\n"{video_audio_ver}"', bg=BLUE, font_size=16
    )
    elements.extend(fq + cv + pv)

    # Hash box
    hash_w, hash_h = 160, 50
    total_row_w = (pv_x + pv_w) - fq_x
    hash_x = fq_x + total_row_w / 2 - hash_w / 2
    hash_y = row_y + bh + 80
    hb = _make_rect(hash_x, hash_y, hash_w, hash_h, label="HASH (SHA)", bg=YELLOW)
    elements.extend(hb)

    for src, sw in [(fq, bw), (cv, bw), (pv, pv_w)]:
        bc = _rect_bottom_center(src[0]["x"], row_y, sw, bh)
        tc = _rect_top_center(hash_x, hash_y, hash_w, hash_h)
        elements.append(_make_arrow(_rect_id(src), _rect_id(hb), bc[0], bc[1], tc[0], tc[1]))

    # Output box
    out_w, out_h = 220, 55
    out_x = hash_x + hash_w / 2 - out_w / 2
    out_y = hash_y + hash_h + 80
    out = _make_rect(
        out_x, out_y, out_w, out_h, label=f'{crop_fq}:audio version\n"{crop_audio_ver}"', bg=GREEN, font_size=16
    )
    elements.extend(out)

    bc = _rect_bottom_center(hash_x, hash_y, hash_w, hash_h)
    tc = _rect_top_center(out_x, out_y, out_w, out_h)
    elements.append(_make_arrow(_rect_id(hb), _rect_id(out), bc[0], bc[1], tc[0], tc[1]))

    return elements


def _build_zone2(ox: float, oy: float, graph: FeatureGraph) -> list[dict[str, Any]]:
    """Zone 2: Feature & Project Version — real features from example-overview."""
    elements: list[dict[str, Any]] = []
    elements.append(_make_title(ox, oy, "Feature & Project Version"))

    features = graph.topological_sort_features()
    n = len(features)
    max_fields = max(len(graph.get_feature_version_by_field(fk)) for fk in features)

    # Column layout
    col_w = 190
    col_gap = 15
    total_cols_w = n * col_w + (n - 1) * col_gap
    cols_start_x = ox + (ZONE_WIDTH - total_cols_w) / 2

    field_bw, field_bh = 170, 38
    row_y = oy + 55

    # Pre-compute aligned y positions based on max field count
    fields_top_y = row_y + 32
    hash1_y = fields_top_y + max_fields * (field_bh + 6) + 15
    hash1_h = 35
    fv_y = hash1_y + hash1_h + 18
    fv_h = 50

    feat_version_rects: list[tuple[list[dict[str, Any]], float, float, float]] = []

    for i, fk in enumerate(features):
        col_x = cols_start_x + i * (col_w + col_gap)
        col_cx = col_x + col_w / 2

        fq_name = fk.to_string()
        version = graph.get_feature_version(fk)
        version_by_field = graph.get_feature_version_by_field(fk)

        # Feature name label
        feat_label = _make_title(col_cx - 80, row_y, fq_name)
        feat_label["fontSize"] = 16
        elements.append(feat_label)

        # Field version boxes
        field_rects: list[list[dict[str, Any]]] = []
        for j, (field_name, field_ver) in enumerate(sorted(version_by_field.items())):
            fy = fields_top_y + j * (field_bh + 6)
            fr = _make_rect(
                col_cx - field_bw / 2, fy, field_bw, field_bh, label=f"{field_name}: {field_ver}", bg=BLUE, font_size=13
            )
            field_rects.append(fr)
            elements.extend(fr)

        # Hash box (aligned across all columns)
        hash_w = 120
        hash_x = col_cx - hash_w / 2
        hb = _make_rect(hash_x, hash1_y, hash_w, hash1_h, label="HASH", bg=YELLOW, font_size=13)
        elements.extend(hb)

        for fr in field_rects:
            bc = _rect_bottom_center(fr[0]["x"], fr[0]["y"], field_bw, field_bh)
            tc = _rect_top_center(hash_x, hash1_y, hash_w, hash1_h)
            elements.append(_make_arrow(_rect_id(fr), _rect_id(hb), bc[0], bc[1], tc[0], tc[1]))

        # Feature version box (aligned across all columns)
        fv_w = 170
        fv_x = col_cx - fv_w / 2
        fv = _make_rect(fv_x, fv_y, fv_w, fv_h, label=f"{fq_name}\n{version}", bg=GREEN, font_size=12)
        elements.extend(fv)
        feat_version_rects.append((fv, fv_x, fv_w, fv_h))

        bc = _rect_bottom_center(hash_x, hash1_y, hash_w, hash1_h)
        tc = _rect_top_center(fv_x, fv_y, fv_w, fv_h)
        elements.append(_make_arrow(_rect_id(hb), _rect_id(fv), bc[0], bc[1], tc[0], tc[1]))

    # Bottom: all feature versions → HASH → project version
    hash2_w, hash2_h = 140, 45
    hash2_x = ox + ZONE_WIDTH / 2 - hash2_w / 2
    hash2_y = fv_y + fv_h + 45
    h2 = _make_rect(hash2_x, hash2_y, hash2_w, hash2_h, label="HASH (SHA)", bg=YELLOW)
    elements.extend(h2)

    for fv, fv_x, fv_w, fv_h_i in feat_version_rects:
        bc = _rect_bottom_center(fv_x, fv_y, fv_w, fv_h_i)
        tc = _rect_top_center(hash2_x, hash2_y, hash2_w, hash2_h)
        elements.append(_make_arrow(_rect_id(fv), _rect_id(h2), bc[0], bc[1], tc[0], tc[1]))

    proj_w, proj_h = 200, 50
    proj_x = ox + ZONE_WIDTH / 2 - proj_w / 2
    proj_y = hash2_y + hash2_h + 40
    pv = _make_rect(proj_x, proj_y, proj_w, proj_h, label=f"Project Version\n{graph.project_version}", bg=GREEN)
    elements.extend(pv)

    bc = _rect_bottom_center(hash2_x, hash2_y, hash2_w, hash2_h)
    tc = _rect_top_center(proj_x, proj_y, proj_w, proj_h)
    elements.append(_make_arrow(_rect_id(h2), _rect_id(pv), bc[0], bc[1], tc[0], tc[1]))

    return elements


def _build_zone3(ox: float, oy: float, graph: FeatureGraph) -> list[dict[str, Any]]:
    """Zone 3: Sample Provenance — concrete: Crop consuming Video."""
    crop_key = FeatureKey("example/crop")
    video_key = FeatureKey("example/video")

    crop_fq = crop_key.to_string()
    video_fq = video_key.to_string()

    crop_defn = graph.get_feature_definition(crop_key)
    video_vbf = graph.get_feature_version_by_field(video_key)
    crop_vbf = graph.get_feature_version_by_field(crop_key)
    crop_version = graph.get_feature_version(crop_key)

    elements: list[dict[str, Any]] = []
    elements.append(_make_title(ox, oy, "Sample Provenance"))

    row_y = oy + 60
    bw, bh = 310, 80

    left_x = ox + 20
    right_x = ox + ZONE_WIDTH - bw - 20

    # Upstream: Video data_version_by_field
    video_fields_str = ", ".join(f'{f}: "{v}"' for f, v in sorted(video_vbf.items()))
    upstream = _make_rect(
        left_x,
        row_y,
        bw,
        bh,
        label=f"{video_fq} (upstream)\ndata_version_by_field\n{{ {video_fields_str} }}",
        bg=BLUE,
        font_size=14,
    )

    # Current: Crop field code versions
    code_ver_str = ", ".join(
        f'{f.key.to_string()}: "{f.code_version}"'
        for f in sorted(crop_defn.spec.fields, key=lambda f: f.key.to_string())
    )
    current = _make_rect(
        right_x,
        row_y,
        bw,
        bh,
        label=f"{crop_fq} (current)\nfield code versions\n{{ {code_ver_str} }}",
        bg=BLUE,
        font_size=14,
    )
    elements.extend(upstream + current)

    # Per-field HASH
    hash1_w, hash1_h = 180, 45
    hash1_x = ox + ZONE_WIDTH / 2 - hash1_w / 2
    hash1_y = row_y + bh + 60
    h1 = _make_rect(hash1_x, hash1_y, hash1_w, hash1_h, label="Per-field HASH", bg=YELLOW)
    elements.extend(h1)

    for src in [upstream, current]:
        bc = _rect_bottom_center(src[0]["x"], row_y, bw, bh)
        tc = _rect_top_center(hash1_x, hash1_y, hash1_w, hash1_h)
        elements.append(_make_arrow(_rect_id(src), _rect_id(h1), bc[0], bc[1], tc[0], tc[1]))

    # provenance_by_field
    crop_prov_str = ", ".join(f'{f}: "{v}"' for f, v in sorted(crop_vbf.items()))
    pbf_w, pbf_h = 340, 55
    pbf_x = ox + ZONE_WIDTH / 2 - pbf_w / 2
    pbf_y = hash1_y + hash1_h + 50
    pbf = _make_rect(
        pbf_x, pbf_y, pbf_w, pbf_h, label=f"provenance_by_field\n{{ {crop_prov_str} }}", bg=GREEN, font_size=14
    )
    elements.extend(pbf)

    bc = _rect_bottom_center(hash1_x, hash1_y, hash1_w, hash1_h)
    tc = _rect_top_center(pbf_x, pbf_y, pbf_w, pbf_h)
    elements.append(_make_arrow(_rect_id(h1), _rect_id(pbf), bc[0], bc[1], tc[0], tc[1]))

    # HASH(all fields)
    hash2_w, hash2_h = 180, 45
    hash2_x = ox + ZONE_WIDTH / 2 - hash2_w / 2
    hash2_y = pbf_y + pbf_h + 50
    h2 = _make_rect(hash2_x, hash2_y, hash2_w, hash2_h, label="HASH (all fields)", bg=YELLOW)
    elements.extend(h2)

    bc = _rect_bottom_center(pbf_x, pbf_y, pbf_w, pbf_h)
    tc = _rect_top_center(hash2_x, hash2_y, hash2_w, hash2_h)
    elements.append(_make_arrow(_rect_id(pbf), _rect_id(h2), bc[0], bc[1], tc[0], tc[1]))

    # provenance (= sample version)
    prov_w, prov_h = 220, 55
    prov_x = ox + ZONE_WIDTH / 2 - prov_w / 2
    prov_y = hash2_y + hash2_h + 50
    prov = _make_rect(prov_x, prov_y, prov_w, prov_h, label=f'provenance\n"{crop_version}"', bg=GREEN, font_size=16)
    elements.extend(prov)

    bc = _rect_bottom_center(hash2_x, hash2_y, hash2_w, hash2_h)
    tc = _rect_top_center(prov_x, prov_y, prov_w, prov_h)
    elements.append(_make_arrow(_rect_id(h2), _rect_id(prov), bc[0], bc[1], tc[0], tc[1]))

    return elements


def _build_zone4(ox: float, oy: float, graph: FeatureGraph) -> list[dict[str, Any]]:
    """Zone 4: Feature Version — upstream field versions + code versions → field versions → feature version.

    Uses example/crop (depends on example/video) as the concrete example.
    """
    crop_key = FeatureKey("example/crop")
    video_key = FeatureKey("example/video")

    crop_fq = crop_key.to_string()
    video_fq = video_key.to_string()

    crop_defn = graph.get_feature_definition(crop_key)
    crop_vbf = graph.get_feature_version_by_field(crop_key)
    crop_version = graph.get_feature_version(crop_key)

    # Collect per-field info: upstream version, code version, resulting field version
    field_infos: list[tuple[str, str, str, str]] = []
    for field_spec in sorted(crop_defn.spec.fields, key=lambda f: f.key.to_string()):
        field_name = field_spec.key.to_string()
        code_ver = field_spec.code_version
        upstream_ver = graph.get_field_version(FQFieldKey(feature=video_key, field=field_spec.key))
        result_ver = crop_vbf[field_name]
        field_infos.append((field_name, upstream_ver, code_ver, result_ver))

    elements: list[dict[str, Any]] = []
    elements.append(_make_title(ox, oy, "Feature Version"))

    n = len(field_infos)
    col_w = 380
    col_gap = 40
    total_w = n * col_w + (n - 1) * col_gap
    cols_start_x = ox + (ZONE_WIDTH - total_w) / 2

    row_y = oy + 60
    input_bw, input_bh = 170, 50
    input_gap = 20

    field_version_rects: list[tuple[list[dict[str, Any]], float, float, float, float]] = []

    for i, (field_name, upstream_ver, code_ver, result_ver) in enumerate(field_infos):
        col_x = cols_start_x + i * (col_w + col_gap)
        col_cx = col_x + col_w / 2

        # Two input boxes side by side: upstream field version + code version
        left_x = col_cx - input_gap / 2 - input_bw
        right_x = col_cx + input_gap / 2

        upstream = _make_rect(
            left_x,
            row_y,
            input_bw,
            input_bh,
            label=f'{video_fq}:{field_name}\n"{upstream_ver}"',
            bg=BLUE,
            font_size=14,
        )
        code = _make_rect(
            right_x,
            row_y,
            input_bw,
            input_bh,
            label=f'code_version\n"{code_ver}"',
            bg=BLUE,
            font_size=14,
        )
        elements.extend(upstream + code)

        # Hash box
        hash_w, hash_h = 120, 40
        hash_x = col_cx - hash_w / 2
        hash_y = row_y + input_bh + 50
        hb = _make_rect(hash_x, hash_y, hash_w, hash_h, label="HASH", bg=YELLOW, font_size=14)
        elements.extend(hb)

        for src, sw in [(upstream, input_bw), (code, input_bw)]:
            bc = _rect_bottom_center(src[0]["x"], row_y, sw, input_bh)
            tc = _rect_top_center(hash_x, hash_y, hash_w, hash_h)
            elements.append(_make_arrow(_rect_id(src), _rect_id(hb), bc[0], bc[1], tc[0], tc[1]))

        # Field version result box
        fv_w, fv_h = 200, 50
        fv_x = col_cx - fv_w / 2
        fv_y = hash_y + hash_h + 50
        fv = _make_rect(
            fv_x,
            fv_y,
            fv_w,
            fv_h,
            label=f'{crop_fq}:{field_name}\n"{result_ver}"',
            bg=GREEN,
            font_size=14,
        )
        elements.extend(fv)
        field_version_rects.append((fv, fv_x, fv_y, fv_w, fv_h))

        bc = _rect_bottom_center(hash_x, hash_y, hash_w, hash_h)
        tc = _rect_top_center(fv_x, fv_y, fv_w, fv_h)
        elements.append(_make_arrow(_rect_id(hb), _rect_id(fv), bc[0], bc[1], tc[0], tc[1]))

    # All field versions → HASH → feature version
    max_fv_bottom = max(y + h for _, _, y, _, h in field_version_rects)

    hash2_w, hash2_h = 140, 45
    hash2_x = ox + ZONE_WIDTH / 2 - hash2_w / 2
    hash2_y = max_fv_bottom + 50
    h2 = _make_rect(hash2_x, hash2_y, hash2_w, hash2_h, label="HASH (SHA)", bg=YELLOW)
    elements.extend(h2)

    for fv, fv_x, fv_y, fv_w, fv_h in field_version_rects:
        bc = _rect_bottom_center(fv_x, fv_y, fv_w, fv_h)
        tc = _rect_top_center(hash2_x, hash2_y, hash2_w, hash2_h)
        elements.append(_make_arrow(_rect_id(fv), _rect_id(h2), bc[0], bc[1], tc[0], tc[1]))

    feat_w, feat_h = 220, 50
    feat_x = ox + ZONE_WIDTH / 2 - feat_w / 2
    feat_y = hash2_y + hash2_h + 50
    feat = _make_rect(
        feat_x,
        feat_y,
        feat_w,
        feat_h,
        label=f'{crop_fq}\n"{crop_version}"',
        bg=GREEN,
        font_size=16,
    )
    elements.extend(feat)

    bc = _rect_bottom_center(hash2_x, hash2_y, hash2_w, hash2_h)
    tc = _rect_top_center(feat_x, feat_y, feat_w, feat_h)
    elements.append(_make_arrow(_rect_id(h2), _rect_id(feat), bc[0], bc[1], tc[0], tc[1]))

    return elements


def _build_separator(x: float, y: float, length: float, *, vertical: bool = False) -> dict[str, Any]:
    """Build a dashed separator line."""
    line_id = _id()
    end_point = [0, length] if vertical else [length, 0]
    return {
        "id": line_id,
        "type": "line",
        "x": x,
        "y": y,
        "width": 0 if vertical else length,
        "height": length if vertical else 0,
        "strokeColor": "#ced4da",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "roughness": 0,
        "opacity": 100,
        "angle": 0,
        "strokeStyle": "dashed",
        "roundness": None,
        "seed": hash(line_id) % (2**31),
        "version": 1,
        "isDeleted": False,
        "boundElements": None,
        "link": None,
        "locked": False,
        "groupIds": [],
        "frameId": None,
        "points": [[0, 0], end_point],
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": None,
        "index": "a0",
        "updated": 1,
    }


def _load_graph() -> FeatureGraph:
    """Load the feature graph from examples/example-overview/."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mx.init(mx.MetaxyConfig(entrypoints=["example_overview.features"], project="example_overview"))
    return FeatureGraph.get()


def generate(graph: FeatureGraph) -> dict[str, Any]:
    """Generate the complete Excalidraw document."""
    elements: list[dict[str, Any]] = []

    # Zone origins (2x2 grid)
    z1_x, z1_y = TOP_LEFT
    z2_x = z1_x + ZONE_WIDTH + ZONE_GAP
    z2_y = z1_y
    z3_x = z1_x
    z3_y = z1_y + ZONE_HEIGHT + ZONE_GAP
    z4_x = z2_x
    z4_y = z3_y

    elements.extend(_build_zone1(z1_x, z1_y, graph))
    elements.extend(_build_zone2(z2_x, z2_y, graph))
    elements.extend(_build_zone3(z3_x, z3_y, graph))
    elements.extend(_build_zone4(z4_x, z4_y, graph))

    # Separator lines between zones
    total_w = 2 * ZONE_WIDTH + ZONE_GAP
    total_h = 2 * ZONE_HEIGHT + ZONE_GAP

    # Horizontal separator
    sep_h_y = z1_y + ZONE_HEIGHT + ZONE_GAP / 2
    elements.append(_build_separator(z1_x, sep_h_y, total_w))

    # Vertical separator
    sep_v_x = z1_x + ZONE_WIDTH + ZONE_GAP / 2
    elements.append(_build_separator(sep_v_x, z1_y, total_h, vertical=True))

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": 20,
            "gridStep": 5,
            "gridModeEnabled": False,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }


def main() -> None:
    graph = _load_graph()
    output_path = (
        Path(__file__).resolve().parent.parent / "docs" / "guide" / "concepts" / "diagrams" / "versioning.excalidraw"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = generate(graph)
    output_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
