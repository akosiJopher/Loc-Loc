"""
Tests for the worker to dashboard connection.

Checks the counts.json merge writing, the per zone freshness check, and
the hold last good count behavior when the camera worker has a hiccup.
The heavy vision libraries are stubbed so this runs on any machine.

Run:  py test_pipeline.py
"""
import importlib
import json
import sys
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

# Stub cv2/ultralytics so camera_worker imports without the vision stack.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_ul = types.ModuleType("ultralytics"); _ul.YOLO = object
sys.modules.setdefault("ultralytics", _ul)

import camera_worker as cw  # noqa: E402
import data_source as ds    # noqa: E402

COUNTS = Path(__file__).parent / "counts.json"
_results = []


def check(name, cond):
    _results.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)


def fresh_ds():
    importlib.reload(ds)
    return ds


def main():
    backup = COUNTS.read_text() if COUNTS.exists() else None
    try:
        cw._CURRENT_MODEL = "yolo26s.pt"

        # --- merge semantics ---
        COUNTS.write_text("{}")
        cw.write_counts("book_common_2", 7, "live")
        cw.write_counts("library_lounge", 3, "live")
        data = json.loads(COUNTS.read_text())
        check("two workers merge into one counts.json",
              data["counts"] == {"book_common_2": 7, "library_lounge": 3})
        check("per-zone metadata written for both zones",
              set(data.get("zones", {})) == {"book_common_2", "library_lounge"})

        d = fresh_ds()
        check("fresh per-zone counts are served",
              d._read_live_counts() == {"book_common_2": 7, "library_lounge": 3})

        # --- per-zone staleness ---
        data["zones"]["library_lounge"]["updated_at"] = (
            datetime.now() - timedelta(seconds=60)).isoformat()
        COUNTS.write_text(json.dumps(data))
        d._LAST_GOOD = {"counts": {}, "ts": 0.0}
        check("stale zone dropped, healthy zone kept",
              d._read_live_counts() == {"book_common_2": 7})

        # --- hold-last-good through a hiccup ---
        for z in data["zones"].values():
            z["updated_at"] = (datetime.now() - timedelta(seconds=60)).isoformat()
        COUNTS.write_text(json.dumps(data))
        check("last good count held while everything is stale",
              d._read_live_counts() == {"book_common_2": 7})
        d._LAST_GOOD["ts"] = time.time() - 999
        check("hold expires after the window",
              d._read_live_counts() == {})

        # --- reconnecting still serves last count ---
        data["zones"]["book_common_2"].update(
            status="reconnecting", updated_at=datetime.now().isoformat())
        COUNTS.write_text(json.dumps(data))
        check("reconnecting worker still serves its last count",
              d._read_live_counts() == {"book_common_2": 7})

        # --- end-to-end ---
        counts = d.get_current_counts()
        check("live zone value flows to get_current_counts",
              counts["book_common_2"] == 7)
        check("every zone within 0..capacity",
              all(0 <= v <= d.ZONES[z]["capacity"] for z, v in counts.items()))
    finally:
        if backup is not None:
            COUNTS.write_text(backup)

    failed = sum(1 for _, ok in _results if not ok)
    print("=" * 48)
    print(f"  RESULT: {len(_results) - failed} passed, {failed} failed")
    print("=" * 48)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
