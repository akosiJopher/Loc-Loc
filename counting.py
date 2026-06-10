"""
LOC-LOC Camera Worker (v2.2.0)

This connects to our Tapo C200 camera using RTSP then runs YOLO with
tracking to count people inside the activity zone. The count is saved
to counts.json and the dashboard reads it from there.

Why tracking and not just detection:
Before we counted the boxes every frame. Problem is the number jumps a
lot when a person moves or gets blocked by another person. With tracking
every person gets an ID that stays with them. A new ID must show up for
a few frames first before we count it (so fake detections are ignored),
and if a counted person disappears for a moment we keep them for a short
grace period (so the count does not drop just because someone moved).

Performance:
- inference runs on a smaller copy of the frame and the fps is capped so
  the CPU is not always at 100%
- a background thread keeps only the newest frame from the camera
- if you have a GPU set device = "cuda" in secrets.toml

Security:
- camera username/password comes from .streamlit/secrets.toml or env
  variables, not hardcoded. The password is hidden in the logs.
- if no credentials are found it refuses to run with the demo password
  unless you set LOCLOC_ALLOW_DEV_FALLBACK=1

To run:
    python camera_worker.py
Press Ctrl+C to stop.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
from ultralytics import YOLO

from counting import (
    PresenceTracker, FallbackSmoother,
    load_roi_zones, ids_inside_zones, reference_points,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    # camera settings
    username: str = "lokasyon"
    password: str = "12345678"
    ip: str = "192.168.254.198"
    port: int = 554
    stream_path: str = "stream1"

    # model / detection settings
    model_path: str = "yolo26s.pt"  # yolo26n.pt = faster, yolo11s.pt = fallback if tracking has issues
    device: str = "cpu"          # "cpu" | "cuda" | "mps"
    confidence: float = 0.5
    iou: float = 0.5
    infer_imgsz: int = 640       # frame is resized to this for detection
    tracker_cfg: str = "bytetrack.yaml"  # switches to locloc_bytetrack.yaml if that file exists

    # stability settings (see counting.py)
    confirm_frames: int = 3
    grace_frames: int = 18       # around 2-3 sec of holding at the fps below

    # timing settings
    target_fps: float = 8.0      # max detection fps
    write_interval_s: float = 1.0
    snapshot_interval_s: float = 0.1      # preview update speed
    roi_reload_s: float = 2.0

    # which zone this camera watches
    zone_id: str = "book_common_2"

    @property
    def rtsp_url(self) -> str:
        # URL-encode credentials so special characters don't break the URL.
        u = quote(self.username, safe="")
        p = quote(self.password, safe="")
        return f"rtsp://{u}:{p}@{self.ip}:{self.port}/{self.stream_path}"

    @property
    def rtsp_url_redacted(self) -> str:
        u = quote(self.username, safe="")
        return f"rtsp://{u}:****@{self.ip}:{self.port}/{self.stream_path}"


def load_config() -> Config:
    """Load settings. Order: defaults, then secrets.toml, then env vars.
    The later one wins if the same setting is in both."""
    cfg = Config()
    _DEV_DEFAULTS = (Config.username, Config.password, Config.ip)

    # 1) secrets.toml
    secrets_file = SCRIPT_DIR / ".streamlit" / "secrets.toml"
    used_secrets = False
    if secrets_file.exists():
        try:
            try:
                import tomllib  # py3.11+
            except ImportError:  # pragma: no cover
                import tomli as tomllib  # type: ignore
            with open(secrets_file, "rb") as f:
                data = tomllib.load(f)
            cam = data.get("camera", {})
            for key in ("username", "password", "ip", "stream_path"):
                if key in cam:
                    setattr(cfg, key, str(cam[key]))
                    used_secrets = True
            if "port" in cam:
                cfg.port = int(cam["port"]); used_secrets = True
            det = data.get("detection", {})
            for key, cast in (
                ("model_path", str), ("device", str), ("confidence", float),
                ("iou", float), ("infer_imgsz", int), ("tracker_cfg", str),
                ("confirm_frames", int), ("grace_frames", int),
                ("target_fps", float), ("zone_id", str),
            ):
                if key in det:
                    setattr(cfg, key, cast(det[key]))
        except Exception as e:
            logging.warning("Could not parse secrets.toml (%s); continuing.", e)

    # 2) environment overrides (LOCLOC_*)
    env_map = {
        "LOCLOC_CAM_USER": ("username", str), "LOCLOC_CAM_PASS": ("password", str),
        "LOCLOC_CAM_IP": ("ip", str), "LOCLOC_CAM_PORT": ("port", int),
        "LOCLOC_STREAM": ("stream_path", str), "LOCLOC_DEVICE": ("device", str),
        "LOCLOC_MODEL": ("model_path", str), "LOCLOC_CONF": ("confidence", float),
        "LOCLOC_IMGSZ": ("infer_imgsz", int), "LOCLOC_CONFIRM": ("confirm_frames", int),
        "LOCLOC_GRACE": ("grace_frames", int), "LOCLOC_FPS": ("target_fps", float),
        "LOCLOC_ZONE": ("zone_id", str),
    }
    for env_key, (attr, cast) in env_map.items():
        if env_key in os.environ and os.environ[env_key].strip():
            try:
                setattr(cfg, attr, cast(os.environ[env_key].strip()))
            except ValueError:
                logging.warning("Bad value for %s; ignoring.", env_key)

    # 3) safety check: dont run with the built in demo password unless
    #    the user really wants to (for local testing only)
    # use our tuned tracker config if the file is there (it remembers
    # tracks longer so people hidden behind others keep their ID)
    if cfg.tracker_cfg == "bytetrack.yaml":
        local_tracker = SCRIPT_DIR / "locloc_bytetrack.yaml"
        if local_tracker.exists():
            cfg.tracker_cfg = str(local_tracker)

    on_defaults = (cfg.username, cfg.password, cfg.ip) == _DEV_DEFAULTS
    allow_dev = os.environ.get("LOCLOC_ALLOW_DEV_FALLBACK") == "1"
    if on_defaults and not (used_secrets or allow_dev):
        logging.error(
            "No camera credentials found (no secrets.toml [camera] and no "
            "LOCLOC_CAM_* env vars). Refusing to use the built-in demo "
            "password. Set credentials in .streamlit/secrets.toml, or set "
            "LOCLOC_ALLOW_DEV_FALLBACK=1 for local testing."
        )
        sys.exit(2)
    return cfg


# ============================================================
# OUTPUT FILES
# ============================================================

STATIC_DIR = SCRIPT_DIR / "static"; STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_FILE     = SCRIPT_DIR / "counts.json"
SNAPSHOT_FILE   = SCRIPT_DIR / "latest_frame.jpg"
SNAPSHOT_STATIC = STATIC_DIR / "latest_frame.jpg"
ROI_CONFIG_FILE = str(SCRIPT_DIR / "roi_config.json")


def roi_path_for_zone(zone_id: str) -> str:
    """If there is a roi file for this specific zone use it, if not use
    the normal roi_config.json. Useful when we add more cameras."""
    per_zone = SCRIPT_DIR / f"roi_config.{zone_id}.json"
    return str(per_zone) if per_zone.exists() else ROI_CONFIG_FILE
FRAME_META_FILE = SCRIPT_DIR / "frame_meta.json"

# model name, saved into counts.json so the dashboard caption shows the
# right model. set once in main()
_CURRENT_MODEL = ""


def write_counts(zone_id: str, count: int, status: str = "live",
                 extra: dict | None = None) -> None:
    """Save the count to counts.json.

    We dont replace the whole file, we read it first and only update our
    own zone. That way if we run 2 or more cameras (one worker per zone)
    they will not erase each other. Each zone also gets its own status
    and updated_at inside "zones" so one camera reconnecting does not
    affect the other one. The old top level keys are still written the
    same so old code that reads them still works.

    Writing is done to a temp file first then renamed, so the file is
    never half written when the dashboard reads it.
    """
    now_iso = datetime.now().isoformat()
    data: dict = {}
    try:
        if OUTPUT_FILE.exists():
            data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
    except (OSError, json.JSONDecodeError, ValueError):
        data = {}

    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    counts[zone_id] = int(count)
    zones_meta = data.get("zones") if isinstance(data.get("zones"), dict) else {}
    meta = {"status": status, "updated_at": now_iso, "model": _CURRENT_MODEL}
    if extra:
        meta.update(extra)
    zones_meta[zone_id] = meta

    out = {
        "updated_at": now_iso,        # legacy top-level fields (back-compat)
        "status": status,
        "counts": counts,
        "model": _CURRENT_MODEL,
        "zones": zones_meta,          # per-zone truth (new)
    }
    tmp = OUTPUT_FILE.with_suffix(f".{zone_id}.tmp")
    try:
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        os.replace(tmp, OUTPUT_FILE)
    except OSError as e:
        logging.warning("Failed to write counts.json: %s", e)


def save_snapshot(frame) -> None:
    """Save a small JPEG of the camera view for the dashboard preview.
    A copy goes to /static because that is the folder streamlit can serve."""
    try:
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        tmp = SNAPSHOT_FILE.with_suffix(".jpg.tmp")
        if cv2.imwrite(str(tmp), frame, params):
            os.replace(tmp, SNAPSHOT_FILE)
        # the dashboard <img> reads this file like 5 times per second.
        # write to temp then rename, or else sometimes the browser reads
        # a half written jpg and the preview shows broken for a moment
        tmp2 = SNAPSHOT_STATIC.with_suffix(".jpg.tmp")
        if cv2.imwrite(str(tmp2), frame, params):
            os.replace(tmp2, SNAPSHOT_STATIC)
    except Exception as e:  # noqa: BLE001 - snapshot is best-effort
        logging.debug("snapshot failed: %s", e)


def save_frame_meta(width: int, height: int) -> None:
    try:
        FRAME_META_FILE.write_text(
            json.dumps({"width": int(width), "height": int(height)}),
            encoding="utf-8",
        )
    except OSError:
        pass


# ============================================================
# OVERLAY (for the dashboard preview)
# ============================================================

def draw_overlay(frame, zones, boxes, ids, inside_flags):
    out = frame.copy()
    h, w = out.shape[:2]
    if zones:
        overlay = out.copy()
        for (x1, y1, x2, y2) in zones:
            p1 = (max(0, min(w - 1, x1)), max(0, min(h - 1, y1)))
            p2 = (max(0, min(w - 1, x2)), max(0, min(h - 1, y2)))
            cv2.rectangle(overlay, p1, p2, (50, 220, 220), -1)
            cv2.rectangle(out, p1, p2, (50, 220, 220), 2)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        inside = inside_flags[i]
        color = (0, 200, 0) if inside else (0, 170, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{ids[i]}" if ids[i] is not None else "?"
        cv2.putText(out, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        bx, by = reference_points(box)[1]
        cv2.circle(out, (int(bx), int(by)), 4, color, -1)
    return out


# ============================================================
# THREADED CAMERA READER
# ============================================================

class CameraReader:
    """Reads camera frames in a background thread and only keeps the
    newest one. Old frames get dropped so the detection is never behind
    the real time."""

    def __init__(self, rtsp_url: str, redacted: str):
        self.rtsp_url = rtsp_url
        self.redacted = redacted
        self.cap = None
        self.latest_frame = None
        self.last_frame_ts = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.read_failures = 0
        self.total_reconnects = 0
        self.state = "starting"  # starting | live | reconnecting | error

    def _open(self):
        # use TCP for RTSP, way more stable on wifi than UDP
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def start(self):
        for attempt in range(1, 6):
            cap = self._open()
            if cap is not None:
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    self.latest_frame = frame
                    self.last_frame_ts = time.time()
                    self.state = "live"
                    self.running = True
                    self.thread = threading.Thread(target=self._loop, daemon=True)
                    self.thread.start()
                    logging.info("Connected on attempt %d.", attempt)
                    return frame
                cap.release()
            logging.warning("Connect attempt %d failed; retrying in 3s...", attempt)
            time.sleep(3)
        self.state = "error"
        return None

    def _loop(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
            except Exception:  # noqa: BLE001
                ok, frame = False, None
            if not ok or frame is None:
                self.read_failures += 1
                if self.read_failures >= 30:
                    self._reconnect()
                else:
                    time.sleep(0.03)
                continue
            self.read_failures = 0
            with self.lock:
                self.latest_frame = frame
                self.last_frame_ts = time.time()

    def _reconnect(self):
        self.state = "reconnecting"
        self.total_reconnects += 1
        logging.warning("Reconnecting (#%d)...", self.total_reconnects)
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass
        for attempt in range(1, 11):
            if not self.running:
                return
            cap = self._open()
            if cap is not None:
                self.cap = cap
                self.read_failures = 0
                self.state = "live"
                logging.info("Reconnected on attempt %d.", attempt)
                return
            logging.warning("Reconnect attempt %d failed; retrying in 5s...", attempt)
            time.sleep(5)
        self.read_failures = 0
        time.sleep(15)

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.last_frame_ts

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass


# ============================================================
# MAIN
# ============================================================

_stop = threading.Event()
def _handle_signal(signum, frame):  # noqa: ARG001
    _stop.set()


def main() -> None:
    from logging.handlers import RotatingFileHandler
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(RotatingFileHandler(
            SCRIPT_DIR / "worker.log", maxBytes=1_000_000, backupCount=3,
            encoding="utf-8",
        ))
    except OSError:
        pass  # cant make a log file, console logging still works
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = load_config()
    logging.info("=" * 56)
    logging.info("LOC-LOC Camera Worker v2.2.0  (tracking based counting)")
    logging.info("Zone: %s | Camera: %s | Device: %s",
                 cfg.zone_id, cfg.rtsp_url_redacted, cfg.device)
    logging.info("Stability: confirm=%d grace=%d | conf=%.2f imgsz=%d fps=%.0f",
                 cfg.confirm_frames, cfg.grace_frames,
                 cfg.confidence, cfg.infer_imgsz, cfg.target_fps)
    logging.info("=" * 56)

    write_counts(cfg.zone_id, 0, status="starting")

    logging.info("[1/3] Loading YOLO model (%s)...", cfg.model_path)
    model = YOLO(cfg.model_path)
    global _CURRENT_MODEL
    _CURRENT_MODEL = Path(cfg.model_path).name

    logging.info("[2/3] Connecting to camera...")
    reader = CameraReader(cfg.rtsp_url, cfg.rtsp_url_redacted)
    first = reader.start()
    if first is None:
        logging.error("Could not connect after 5 attempts. Test this URL in "
                      "VLC: %s", cfg.rtsp_url_redacted)
        write_counts(cfg.zone_id, 0, status="camera_error")
        return
    fh, fw = first.shape[:2]
    save_frame_meta(fw, fh)
    logging.info("      Frame size: %d x %d", fw, fh)

    logging.info("[3/3] Detection loop running. Ctrl+C to stop.")
    roi_file = roi_path_for_zone(cfg.zone_id)
    zones = load_roi_zones(roi_file)
    logging.info("      %s", f"{len(zones)} ROI zone(s) loaded." if zones
                 else "No ROI zones - counting everyone in the frame.")

    tracker = PresenceTracker(cfg.confirm_frames, cfg.grace_frames)
    fallback = FallbackSmoother(window=15)
    last_count = 0
    last_written = None          # last count actually written -> event-driven writes
    min_period = 1.0 / max(0.5, cfg.target_fps)
    t_write = t_snap = t_roi = t_stale_write = 0.0
    fps_ema = 0.0                # observability: actual achieved inference fps
    last_wh = (fw, fh)

    try:
        while not _stop.is_set():
            loop_start = time.time()
            frame, frame_ts = reader.get_latest()
            if frame is None:
                if reader.state == "reconnecting":
                    write_counts(cfg.zone_id, last_count, status="reconnecting")
                time.sleep(0.05)
                continue

            # if the frames are old (camera problem) skip the detection.
            # no point running YOLO again on the same frozen frame, it
            # just wastes CPU and the status would lie that we are live
            if frame_ts and (time.time() - frame_ts) > 5.0:
                if time.time() - t_stale_write >= 1.0:
                    write_counts(cfg.zone_id, last_count, status="reconnecting")
                    t_stale_write = time.time()
                time.sleep(0.2)
                continue

            # camera can change resolution after a reconnect. keep
            # frame_meta.json updated because the ROI editor uses it
            wh = (frame.shape[1], frame.shape[0])
            if wh != last_wh:
                save_frame_meta(*wh)
                logging.info("Frame size changed: %dx%d -> %dx%d",
                             last_wh[0], last_wh[1], wh[0], wh[1])
                last_wh = wh

            now = time.time()
            if now - t_roi >= cfg.roi_reload_s:
                new_zones = load_roi_zones(roi_file)
                if new_zones != zones:
                    zones = new_zones
                    logging.info("ROI updated: %d zone(s)", len(zones))
                t_roi = now

            # run YOLO with tracking (this is the important part)
            try:
                results = model.track(
                    frame, persist=True, classes=[0],
                    conf=cfg.confidence, iou=cfg.iou,
                    imgsz=cfg.infer_imgsz, tracker=cfg.tracker_cfg,
                    device=cfg.device, verbose=False,
                )
            except Exception as e:  # noqa: BLE001
                logging.warning("track() failed: %s", e)
                time.sleep(0.1)
                continue

            boxes, ids = [], []
            try:
                r = results[0] if results else None
                if r is not None and r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy().tolist()
                    if r.boxes.id is not None:
                        ids = [int(i) for i in r.boxes.id.cpu().numpy().tolist()]
                    else:
                        ids = [None] * len(boxes)
            except Exception as e:  # noqa: BLE001
                logging.debug("box extraction failed: %s", e)

            inside_ids, raw_inside = ids_inside_zones(boxes, ids, zones)
            inside_flags = [
                (box_in_zones_safe(b, zones)) for b in boxes
            ]

            if any(i is not None for i in ids):
                # Primary path: stable, ID-based count.
                stable = tracker.update(inside_ids)
            else:
                # Tracker warm-up / no IDs yet: fall back to smoothed raw count.
                stable = fallback.update(raw_inside)
            last_count = stable

            # write right away when the count changes. the dashboard will
            # see it on its next refresh so people entering or leaving
            # show up fast, no extra 1 second delay on top
            if stable != last_written:
                write_counts(cfg.zone_id, stable, status="live",
                             extra={"fps": round(fps_ema, 1),
                                    "reconnects": reader.total_reconnects})
                last_written = stable
                t_write = now

            # snapshot (throttled)
            if now - t_snap >= cfg.snapshot_interval_s:
                try:
                    annotated = draw_overlay(frame, zones, boxes, ids, inside_flags)
                    save_snapshot(annotated)
                except Exception as e:  # noqa: BLE001
                    logging.debug("overlay/snapshot failed: %s", e)
                t_snap = now

            # heartbeat write every second even if no change, so the
            # dashboard can tell apart "no change" vs "worker is dead"
            if now - t_write >= cfg.write_interval_s:
                write_counts(cfg.zone_id, stable, status="live",
                             extra={"fps": round(fps_ema, 1),
                                    "reconnects": reader.total_reconnects})
                last_written = stable
                t_write = now
                mode = f"ROI({len(zones)})" if zones else "ALL"
                logging.info("[%s] det=%d inside=%d stable=%d%s",
                             mode, len(boxes), raw_inside, stable,
                             f" reconnects={reader.total_reconnects}"
                             if reader.total_reconnects else "")

            # fps limit + measure the real fps we are getting
            elapsed = time.time() - loop_start
            inst = 1.0 / max(elapsed, 1e-6)
            fps_ema = inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * min(inst, cfg.target_fps))
            if elapsed < min_period:
                time.sleep(min_period - elapsed)
    finally:
        reader.stop()
        write_counts(cfg.zone_id, last_count, status="stopped")
        logging.info("Worker stopped. Reconnects this run: %d",
                     reader.total_reconnects)


def box_in_zones_safe(box, zones):
    from counting import box_in_zones
    try:
        return box_in_zones(box, zones)
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
