"""Data collection, heatmap, and RL self-verification storage."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .utils import get_evidence_filename, load_all_evidence


def evidence_static_prefix() -> str:
    """URL prefix for serving evidence files from the browser."""
    if Path("/data/evidence").is_dir():
        return "/view-evidence"
    return "/evidence"


def training_static_prefix() -> str:
    """URL prefix for serving RL training files from the browser."""
    if Path("/data/rl_training").is_dir():
        return "/view-training"
    return "/training-data"


def _is_image_evidence_record(record: Dict[str, Any]) -> bool:
    """True for image pipeline records (timestamp + gps), not stop-line video JSON."""
    return (
        isinstance(record.get("timestamp"), str)
        and isinstance(record.get("gps"), (list, tuple))
        and len(record.get("gps", [])) == 2
    )


def _record_id(record: Dict[str, Any]) -> str:
    return get_evidence_filename(record["timestamp"])


def _image_filename(record: Dict[str, Any]) -> str:
    return f"{_record_id(record)}.jpg"


def _json_filename(record: Dict[str, Any]) -> str:
    return f"{_record_id(record)}.json"


def _parse_gps(gps: Any) -> Tuple[float, float]:
    if isinstance(gps, (list, tuple)) and len(gps) == 2:
        return float(gps[0]), float(gps[1])
    return 0.0, 0.0


def get_system_model_outputs(record: Dict[str, Any]) -> Dict[str, Any]:
    plates = record.get("plates") or []
    violations = record.get("violations") or []
    plate_texts = [p.get("text", "") for p in plates if isinstance(p, dict)]

    return {
        "vehicle_detected": len(violations) > 0,
        "license_plate_detected": len(plates) > 0,
        "ocr_detected": any(str(t).strip() for t in plate_texts),
        "violation_types": [
            v.get("type", "unknown") for v in violations if isinstance(v, dict)
        ],
        "plate_texts": plate_texts,
    }


def enrich_evidence_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add browser-accessible URLs and normalized fields to an evidence record."""
    prefix = evidence_static_prefix()
    record_id = _record_id(record)
    lat, lon = _parse_gps(record.get("gps"))

    enriched = dict(record)
    enriched["id"] = record_id
    enriched["gps_lat"] = lat
    enriched["gps_lon"] = lon
    enriched["image_filename"] = _image_filename(record)
    enriched["json_filename"] = _json_filename(record)
    enriched["image_url"] = f"{prefix}/{_image_filename(record)}"
    enriched["json_url"] = f"{prefix}/{_json_filename(record)}"
    enriched["system_outputs"] = get_system_model_outputs(record)
    return enriched


def load_collection_violations(
    include_zero_gps: bool = True,
) -> List[Dict[str, Any]]:
    """Load image evidence records with URLs for frontend collection / violations view."""
    records = []
    for record in load_all_evidence():
        if not _is_image_evidence_record(record):
            continue
        lat, lon = _parse_gps(record.get("gps"))
        if not include_zero_gps and lat == 0.0 and lon == 0.0:
            continue
        records.append(enrich_evidence_record(record))
    return records


def load_heatmap_points(include_zero_gps: bool = False) -> List[Dict[str, Any]]:
    """Return GPS points with violation metadata for map heatmap rendering."""
    points = []
    for record in load_collection_violations(include_zero_gps=include_zero_gps):
        violation_types = [
            v.get("type", "unknown") for v in record.get("violations", [])
        ]
        points.append(
            {
                "id": record["id"],
                "lat": record["gps_lat"],
                "lon": record["gps_lon"],
                "timestamp": record["timestamp"],
                "weight": max(len(violation_types), 1),
                "violation_types": violation_types,
                "violation_count": len(violation_types),
                "image_url": record["image_url"],
            }
        )
    return points


def find_evidence_by_id(violation_id: str) -> Optional[Dict[str, Any]]:
    """Find an image evidence record by timestamp stem / id."""
    target = violation_id.strip()
    if target.endswith(".json"):
        target = target[:-5]
    if target.endswith(".jpg"):
        target = target[:-4]

    for record in load_all_evidence():
        if not _is_image_evidence_record(record):
            continue
        if _record_id(record) == target:
            return record
    return None


def _training_category(human_labels: Dict[str, Any], violation_confirmed: bool) -> str:
    """Route verified samples to confirmed vs corrections folders."""
    if not violation_confirmed:
        return "corrections"

    for label in human_labels.values():
        if isinstance(label, dict) and label.get("correct") is False:
            return "corrections"
    return "confirmed"


def verify_violation_for_training(
    violation_id: str,
    violation_confirmed: bool,
    ocr: Dict[str, Any],
    license_plate: Dict[str, Any],
    vehicle: Dict[str, Any],
    annotation_notes: Optional[str] = None,
    helmet: Optional[Dict[str, Any]] = None,
    seatbelt: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Copy a verified violation into RL training storage.

    Original evidence stays in place so existing /violations frontend keeps working.
    """
    source = find_evidence_by_id(violation_id)
    if source is None:
        return False, {"error": f"Violation not found: {violation_id}"}

    record_id = _record_id(source)
    system_outputs = get_system_model_outputs(source)
    human_labels = {
        "ocr": ocr,
        "license_plate": license_plate,
        "vehicle": vehicle,
    }
    if helmet is not None:
        human_labels["helmet"] = helmet
    if seatbelt is not None:
        human_labels["seatbelt"] = seatbelt

    category = _training_category(human_labels, violation_confirmed)
    category_dir = Path(Config.rl_category_folder(category))
    category_dir.mkdir(parents=True, exist_ok=True)

    source_image = Path(Config.EVIDENCE_FOLDER) / _image_filename(source)
    if not source_image.is_file():
        return False, {"error": f"Evidence image not found: {source_image.name}"}

    dest_image = category_dir / source_image.name
    shutil.copy2(source_image, dest_image)

    verified_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    training_prefix = training_static_prefix()
    training_record = {
        "id": record_id,
        "source_violation_id": record_id,
        "verified_at": verified_at,
        "violation_confirmed": violation_confirmed,
        "timestamp": source.get("timestamp"),
        "gps": source.get("gps"),
        "original_evidence": source,
        "system_outputs": system_outputs,
        "human_labels": human_labels,
        "annotation_notes": annotation_notes or "",
        "category": category,
        "has_model_errors": category == "corrections",
        "image_filename": dest_image.name,
        "image_url": f"{training_prefix}/{category}/{dest_image.name}",
        "json_url": f"{training_prefix}/{category}/{record_id}.json",
    }

    dest_json = category_dir / f"{record_id}.json"
    with open(dest_json, "w", encoding="utf-8") as handle:
        json.dump(training_record, handle, indent=2)

    return True, training_record


def _load_training_records_in_folder(folder: Path) -> List[Dict[str, Any]]:
    records = []
    if not folder.is_dir():
        return records

    prefix = training_static_prefix()
    for json_file in sorted(folder.glob("*.json"), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            record.setdefault("json_url", f"{prefix}/{folder.name}/{json_file.name}")
            records.append(record)
        except Exception:
            continue
    return records


def load_all_training_data(
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load RL training records from confirmed and/or corrections folders."""
    if category in ("confirmed", "corrections"):
        return _load_training_records_in_folder(
            Path(Config.rl_category_folder(category))
        )

    confirmed = _load_training_records_in_folder(
        Path(Config.rl_category_folder("confirmed"))
    )
    corrections = _load_training_records_in_folder(
        Path(Config.rl_category_folder("corrections"))
    )
    return confirmed + corrections


def get_training_stats() -> Dict[str, Any]:
    """Aggregate RL dataset counts and retraining readiness."""
    confirmed = _load_training_records_in_folder(
        Path(Config.rl_category_folder("confirmed"))
    )
    corrections = _load_training_records_in_folder(
        Path(Config.rl_category_folder("corrections"))
    )

    model_error_counts = {
        "ocr": 0,
        "license_plate": 0,
        "vehicle": 0,
        "helmet": 0,
        "seatbelt": 0,
    }

    for record in corrections:
        for model_name, label in (record.get("human_labels") or {}).items():
            if model_name in model_error_counts and isinstance(label, dict):
                if label.get("correct") is False:
                    model_error_counts[model_name] += 1

    threshold = Config.RL_RETRAIN_THRESHOLD
    total_corrections = len(corrections)

    return {
        "confirmed_count": len(confirmed),
        "corrections_count": total_corrections,
        "total_training_samples": len(confirmed) + total_corrections,
        "model_error_counts": model_error_counts,
        "retrain_threshold": threshold,
        "ready_for_retraining": total_corrections >= threshold,
        "corrections_until_retrain": max(0, threshold - total_corrections),
        "training_folder": Config.RL_TRAINING_FOLDER,
        "categories": {
            "confirmed": Config.rl_category_folder("confirmed"),
            "corrections": Config.rl_category_folder("corrections"),
        },
    }
