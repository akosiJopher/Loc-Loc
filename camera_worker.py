import base64
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# import the drawable canvas here at startup so the ROI editor opens fast
# the first time. if the package is missing the dialog shows an error
try:
    from streamlit_drawable_canvas import st_canvas as _st_canvas
except Exception:
    _st_canvas = None

# note: older streamlit versions needed a patch here for the drawable
# canvas image_to_url problem. with streamlit 1.58+ and the canvas -fix
# package 0.9.8+ it works directly so no patch needed anymore.

from data_source import (
    ZONES, DEFAULT_BOOKINGS,
    get_current_counts, get_status, get_status_emoji,
    get_smart_suggestion, get_historical_data, get_peak_advisory,
    get_current_booking, get_upcoming_bookings,
)

st.set_page_config(page_title="LOC-LOC", page_icon="📍", layout="wide", initial_sidebar_state="collapsed")

# --- Paths & constants ---
PROJECT_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = PROJECT_DIR / "latest_frame.jpg"
STATIC_SNAPSHOT_PATH = PROJECT_DIR / "static" / "latest_frame.jpg"
ROI_PATH = PROJECT_DIR / "roi_config.json"
COUNTS_PATH = PROJECT_DIR / "counts.json"
FRAME_META_PATH = PROJECT_DIR / "frame_meta.json"
LOGO_PATH = PROJECT_DIR / "logo.png"

DEFAULT_FRAME_W, DEFAULT_FRAME_H = 1920, 1080
LIVE_DATA_MAX_AGE_SECONDS = 10


def safe_mtime(path):
    """Get the file modified time without crashing. The camera worker
    replaces the frame file many times per second, so if we do exists()
    then stat() there is a tiny moment where the file is gone and stat()
    crashes. After hours of running that moment eventually happens. This
    just returns None instead."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def resolve_snapshot_path():
    """Return the freshest existing snapshot, or None.

    The worker saves latest_frame.jpg in the project folder and also a
    copy inside /static (that one is for the live <img>). We check both
    and take the newest one, so the ROI editor always finds an image as
    long as the live feed has one.
    """
    best, best_t = None, -1.0
    for p in (SNAPSHOT_PATH, STATIC_SNAPSHOT_PATH):
        t = safe_mtime(p)
        if t is not None and t > best_t:
            best, best_t = p, t
    return best

# Single source of truth: status -> theme. Used by cards and the map.
STATUS_THEME = {
    "AVAILABLE": {"main": "#10b981", "bg": "#ecfdf5", "text": "#065f46", "pill": "pg"},
    "BUSY":      {"main": "#f59e0b", "bg": "#fffbeb", "text": "#92400e", "pill": "po"},
    "FULL":      {"main": "#ef4444", "bg": "#fef2f2", "text": "#991b1b", "pill": "pr"},
}
DEFAULT_THEME = {"main": "#94a3b8", "bg": "#f8fafc", "text": "#475569", "pill": "pg"}

# Camera worker status -> pill colors + label.
WORKER_STATUS_PILLS = {
    "live":         ("#ecfdf5", "#a7f3d0", "#065f46", "🟢 Camera worker online - updating live"),
    "starting":     ("#fffbeb", "#fde68a", "#92400e", "🟡 Camera worker starting up..."),
    "connecting":   ("#fffbeb", "#fde68a", "#92400e", "🟡 Camera worker starting up..."),
    "reconnecting": ("#fffbeb", "#fde68a", "#92400e", "🔄 Reconnecting to camera..."),
    "camera_error": ("#fef2f2", "#fecaca", "#991b1b", "🔴 Camera connection failed - check network and RTSP URL"),
    "stale":        ("#fef2f2", "#fecaca", "#991b1b", "🔴 Feed is stale - worker may be frozen"),
    "offline":      ("#f1f5f9", "#cbd5e1", "#475569", "⚪ Camera worker offline"),
}


# --- Helpers ---
def read_json_safe(path, default=None):
    """Read JSON returning `default` on any IO/parse error. Stable for live files."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return default if default is not None else {}


def write_json_atomic(path, data):
    """Write via temp + os.replace so a half-written file never appears on disk."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def status_theme(status):
    return STATUS_THEME.get(status, DEFAULT_THEME)


def render_status_pill(bg, border, color, label):
    """Shared markup for camera worker / system status pills.
    Uses inline color (legacy) but wraps in a known class so dark-mode
    CSS can override the background reliably regardless of source color."""
    # Map source bg to a semantic CSS class so dark mode picks it up.
    color_map = {
        "#ecfdf5": "status-pill-success",
        "#fffbeb": "status-pill-warning",
        "#fef2f2": "status-pill-error",
        "#f1f5f9": "status-pill-neutral",
    }
    cls = color_map.get(bg, "status-pill-neutral")
    return (
        f'<div class="status-pill {cls}" style="background:{bg};border:1px solid {border};'
        f'color:{color};">{label}</div>'
    )


def get_worker_status():
    """Read counts.json and decide live/stale/offline. Used by the admin Camera tab."""
    if not COUNTS_PATH.exists():
        return "offline", None
    data = read_json_safe(COUNTS_PATH)
    status = data.get("status", "offline")
    updated_at = data.get("updated_at")
    if updated_at:
        try:
            age = (datetime.now() - datetime.fromisoformat(updated_at)).total_seconds()
            if age > LIVE_DATA_MAX_AGE_SECONDS:
                status = "stale"
        except ValueError:
            pass
    return status, updated_at


def get_detection_label():
    """Caption for the live feed. Reads the model the worker recorded in
    counts.json so it stays correct no matter which model is configured
    (yolo26s / yolo26n / yolo11s / ...). Falls back to the current default."""
    data = read_json_safe(COUNTS_PATH, default={})
    raw = str(data.get("model", "") or "")
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".pt"):
        name = name[:-3]
    if name.lower().startswith("yolo"):
        pretty = "YOLO" + name[4:]
    else:
        pretty = name or "YOLO26s"
    return f"RTSP source: Tapo C200 · {pretty} person detection"


def get_booking_state(today_str):
    """Returns the three booking buckets used by both card variants."""
    current = get_current_booking(st.session_state.bookings)
    upcoming_all = get_upcoming_bookings(st.session_state.bookings, limit=10)
    today_upcoming = [b for b in upcoming_all if b.get("date") == today_str]
    future_upcoming = [b for b in upcoming_all if b.get("date") != today_str]
    return current, today_upcoming, future_upcoming


@st.cache_data(ttl=60)
def get_cached_historical():
    """Trend chart data barely changes, cache it for 60s so the chart is
    not rebuilt on every interaction."""
    return get_historical_data()


@st.fragment(run_every=1.5)
def render_camera_status_fragment():
    """Refreshes the worker status pill and mirrors the worker's snapshot
    into /static so the JS-driven <img> below has something fresh to load.
    Doing the mirror here means the dashboard works even if the worker
    isn't writing to /static directly."""
    # copy latest_frame.jpg to static/latest_frame.jpg but only when the
    # source is actually newer, so we dont copy the same file every tick
    static_path = PROJECT_DIR / "static" / "latest_frame.jpg"
    try:
        static_path.parent.mkdir(exist_ok=True)
        if SNAPSHOT_PATH.exists():
            src_mtime = SNAPSHOT_PATH.stat().st_mtime
            dst_mtime = static_path.stat().st_mtime if static_path.exists() else 0
            if src_mtime > dst_mtime + 0.1:
                import shutil
                tmp_path = static_path.with_suffix(".jpg.tmp")
                shutil.copy2(SNAPSHOT_PATH, tmp_path)
                os.replace(tmp_path, static_path)
    except OSError:
        pass

    worker_status, _ = get_worker_status()
    pill_args = WORKER_STATUS_PILLS.get(worker_status, WORKER_STATUS_PILLS["offline"])
    st.markdown(render_status_pill(*pill_args), unsafe_allow_html=True)
    if worker_status == "offline":
        st.caption("Run `python camera_worker.py` in a terminal to start live detection.")


# --- Live refresh ---
# The dashboard's dynamic sections live inside an st.fragment that ticks
# on its own. The rest of the page never reruns unless the user interacts
# with it, which kills the page-wide flicker the old full-rerun approach
# caused. The ROI editor sits outside the fragment, so its canvas isn't
# reset by background ticks either.
st.session_state.setdefault("show_roi_editor", False)

# --- Session state defaults ---
st.session_state.setdefault("bookings", [dict(b) for b in DEFAULT_BOOKINGS])
st.session_state.setdefault("booking_counter", len(DEFAULT_BOOKINGS))
st.session_state.setdefault("show_details_toggle", False)
# Admin password - read from .streamlit/secrets.toml so credentials don't
# live in source code. Falls back to the hardcoded default if secrets.toml
# is missing (convenient for local dev, remove fallback before deploying).
def _load_admin_password():
    try:
        return st.secrets["admin"]["password"]
    except Exception:
        return "lokasyon2026"
st.session_state.setdefault("admin_password", _load_admin_password())
st.session_state.setdefault("auth_failures", 0)
st.session_state.setdefault("auth_locked_until", 0.0)

AUTH_MAX_ATTEMPTS = 5      # wrong tries before a lockout
AUTH_LOCKOUT_SECONDS = 60  # how long the panel stays locked


def check_admin_password(candidate: str) -> bool:
    """Constant-time password check (hmac.compare_digest) so the comparison
    itself can't leak which prefix of the password was right."""
    import hmac
    return hmac.compare_digest(
        str(candidate).encode("utf-8"),
        str(st.session_state.admin_password).encode("utf-8"),
    )
st.session_state.setdefault("booking_version", 0)


# --- ROI editor dialog ---
@st.dialog("Edit Activity Zones", width="large")
def roi_editor_dialog():
    """Modal for drawing Activity Zones with click-and-drag."""
    current_zones = read_json_safe(ROI_PATH, default={}).get("zones", [])

    meta = read_json_safe(FRAME_META_PATH, default={})
    frame_w = int(meta.get("width", DEFAULT_FRAME_W))
    frame_h = int(meta.get("height", DEFAULT_FRAME_H))

    if _st_canvas is None:
        st.error("⚠️ Requires `streamlit-drawable-canvas-fix`.")
        st.code("python -m pip install streamlit-drawable-canvas-fix")
        return
    st_canvas = _st_canvas
    from PIL import Image

    snap_path = resolve_snapshot_path()
    if snap_path is None:
        st.warning("Start the camera worker first - the editor needs a snapshot to draw on.")
        return

    # The worker rewrites the snapshot every fraction of a second, so we may
    # catch it mid-write. A short retry covers that window.
    bg_img = None
    for _ in range(3):
        try:
            bg_img = Image.open(snap_path)
            bg_img.load()
            break
        except (OSError, IOError):
            time.sleep(0.1)
            bg_img = None

    if bg_img is None:
        st.warning("Could not load camera snapshot. Close this dialog and try again.")
        return

    try:
        bg_w, bg_h = bg_img.size
        canvas_display_w = min(bg_w, 800)
        canvas_display_h = int(bg_h * (canvas_display_w / bg_w))
        scale_x = canvas_display_w / frame_w
        scale_y = canvas_display_h / frame_h

        # Pre-load saved zones onto the canvas as fabric rectangles
        initial_objects = [
            {
                "type": "rect",
                "version": "4.4.0",
                "left": z.get("x1", 0) * scale_x,
                "top": z.get("y1", 0) * scale_y,
                "width": (z.get("x2", 0) - z.get("x1", 0)) * scale_x,
                "height": (z.get("y2", 0) - z.get("y1", 0)) * scale_y,
                "fill": "rgba(20, 184, 166, 0.25)",
                "stroke": "rgb(13, 148, 136)",
                "strokeWidth": 2,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
            }
            for z in current_zones
        ]

        draw_mode = st.radio(
            "Mode",
            ["Draw new", "Edit / delete"],
            key="roi_dialog_mode",
            horizontal=True,
        )
        is_draw_mode = draw_mode == "Draw new"
        canvas_mode = "rect" if is_draw_mode else "transform"

        if is_draw_mode:
            st.caption(
                "✏️ **Draw mode** - Click and drag on the image to create rectangles. "
                "Each rectangle becomes an Activity Zone where people are counted. "
                "Use the toolbar's undo/redo if you make a mistake."
            )
        else:
            st.caption(
                "🛠️ **Edit / delete mode** - Click a rectangle on the image to select it "
                "(resize handles will appear). To DELETE: click the rectangle to select it, "
                "then click the 🗑️ trash icon in the toolbar, or use the Delete buttons "
                "next to each saved zone below."
            )

        canvas_kwargs = dict(
            fill_color="rgba(20, 184, 166, 0.25)",
            stroke_color="rgb(13, 148, 136)",
            stroke_width=2,
            background_image=bg_img,
            update_streamlit=True,
            height=canvas_display_h,
            width=canvas_display_w,
            drawing_mode=canvas_mode,
            display_toolbar=True,
            key="roi_dialog_canvas",
        )
        # Empty initial_drawing causes a blank render on some canvas versions
        if initial_objects:
            canvas_kwargs["initial_drawing"] = {"version": "4.4.0", "objects": initial_objects}

        try:
            canvas_result = st_canvas(**canvas_kwargs)
        except Exception as err:
            import traceback
            st.error(f"Canvas failed to render: {err}")
            st.code(traceback.format_exc())
            canvas_result = None

        st.caption(f"Canvas: {canvas_display_w}×{canvas_display_h} · Source frame: {frame_w}×{frame_h}")

        # Saved-zones list with delete buttons. Hidden in Draw mode so each
        # mode has a single, clear purpose.
        if not is_draw_mode:
            if current_zones:
                st.markdown("**Saved zones:**")
                for i, z in enumerate(current_zones):
                    cols = st.columns([1, 5, 1])
                    with cols[0]:
                        st.markdown(
                            f'<div style="padding:6px 10px;background:#14b8a6;color:white;border-radius:4px;'
                            f'font-family:Atkinson Hyperlegible Next,sans-serif;font-size:0.85rem;font-weight:700;text-align:center;">'
                            f'Zone {i+1}</div>',
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        w = z.get("x2", 0) - z.get("x1", 0)
                        h = z.get("y2", 0) - z.get("y1", 0)
                        st.markdown(
                            f'<div class="zone-coord">'
                            f'({z.get("x1", 0)}, {z.get("y1", 0)}) - size {w}x{h}</div>',
                            unsafe_allow_html=True,
                        )
                    with cols[2]:
                        if st.button("🗑️ Delete", key=f"del_saved_zone_{i}", width='stretch', type="secondary"):
                            try:
                                write_json_atomic(ROI_PATH, {"zones": [nz for j, nz in enumerate(current_zones) if j != i]})
                                st.success(f"Deleted Zone {i+1}.")
                                st.rerun()
                            except OSError as ex:
                                st.error(f"Failed to save: {ex}")
                st.caption("Deletes apply immediately and remove the zone from the saved config.")
            else:
                st.info("No saved zones yet. Switch to **Draw new** mode to add one.")
        else:
            if current_zones:
                st.caption(
                    f"💡 You currently have **{len(current_zones)} saved zone(s)**. "
                    f"To remove or resize them, switch to **Edit / delete** mode."
                )
            else:
                st.caption("No zones saved yet. Draw rectangles above and click Save Zones.")

        col_save, col_close = st.columns(2)
        with col_save:
            if st.button("Save Zones", type="primary", width='stretch', key="roi_dialog_save"):
                if canvas_result is None or canvas_result.json_data is None or not canvas_result.json_data.get("objects"):
                    try:
                        write_json_atomic(ROI_PATH, {"zones": []})
                        st.success("Saved (no zones - counting everyone).")
                        st.session_state.show_roi_editor = False
                        st.rerun()
                    except OSError as ex:
                        st.error(f"Failed to save: {ex}")
                else:
                    new_zones = []
                    for obj in canvas_result.json_data["objects"]:
                        if obj.get("type") != "rect":
                            continue
                        left = float(obj.get("left", 0))
                        top = float(obj.get("top", 0))
                        w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                        h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
                        x1 = max(0, min(frame_w, int(left / scale_x)))
                        y1 = max(0, min(frame_h, int(top / scale_y)))
                        x2 = max(0, min(frame_w, int((left + w) / scale_x)))
                        y2 = max(0, min(frame_h, int((top + h) / scale_y)))
                        if x1 != x2 and y1 != y2:
                            new_zones.append({
                                "x1": min(x1, x2), "y1": min(y1, y2),
                                "x2": max(x1, x2), "y2": max(y1, y2),
                            })
                    try:
                        write_json_atomic(ROI_PATH, {"zones": new_zones})
                        st.success(f"Saved {len(new_zones)} zone(s). Worker will apply within 2 seconds.")
                        st.session_state.show_roi_editor = False
                        st.rerun()
                    except OSError as ex:
                        st.error(f"Failed to save: {ex}")

        with col_close:
            if st.button("Close without saving", width='stretch', key="roi_dialog_close",
                         help="Cancels any changes. Previously saved zones remain unchanged."):
                st.session_state.show_roi_editor = False
                st.rerun()

    except Exception as ex:
        st.error(f"Failed to load zone editor: {ex}")


if st.session_state.show_roi_editor:
    roi_editor_dialog()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] { 
    background-color: #f7f4ec !important;
    color: #0f172a !important;
    font-family: Atkinson Hyperlegible Next, sans-serif;
}

#MainMenu, footer { display: none !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    display: none !important; 
}
/* Hide the default X close button on st.dialog modals - we provide our own */
[data-testid="stModal"] button[kind="header"],
[data-testid="stModal"] [aria-label="Close"],
div[role="dialog"] button[aria-label="Close"] { display: none !important; }

.block-container { 
    padding-top: 0 !important; 
    padding-bottom: 2rem !important; 
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1100px !important; 
}

/* Streamlit gives every element-container a default bottom margin/gap.
   Our header is rendered via components.html with height=0, but its
   wrapping element-container still reserves that gap, which adds
   ~16px of phantom space between the header and the clock. Zero it
   out for any element-container that contains a 0-height iframe. */
[data-testid="element-container"]:has(iframe[height="0"]),
[data-testid="stElementContainer"]:has(iframe[height="0"]),
[data-testid="stIFrame"]:has(iframe[height="0"]) {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
    line-height: 0 !important;
    /* take it out of the flex flow. even a 0 height element still gets
       the flex gap around it, that was the mystery space between the
       header and the clock */
    position: absolute !important;
    width: 0 !important;
    overflow: hidden !important;
}
[data-testid="element-container"] > iframe[height="0"] {
    display: block !important;
    height: 0 !important;
    border: none !important;
}

.clock { font-family:'JetBrains Mono',monospace; font-size:0.76rem; color:#7a1420 !important; margin:0 0 4px 0; font-weight:700; text-transform:uppercase; letter-spacing:1.4px; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* bouncing arrow on the map, appears over the highlighted zone after
   you tap a card. the bounce animation is on an inner group so the
   outer transform can position and flip it */
@keyframes locArrowBounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-11px); }
}
.loc-arrow { pointer-events: none; }
.loc-arrow-inner { animation: locArrowBounce 0.9s cubic-bezier(0.45, 0, 0.55, 1) infinite; }
.loc-arrow polygon { fill: #7a1420; stroke: #fffdf9; stroke-width: 2.5; stroke-linejoin: round; }
.loc-arrow ellipse { fill: #1c1917; opacity: 0.16; }
html[data-theme="dark"] .loc-arrow polygon { fill: #fca5a5; stroke: #0f172a; }
html[data-theme="dark"] .loc-arrow ellipse { fill: #000000; opacity: 0.35; }

/* make the vertical gaps smaller, streamlit default 1rem gap made the
   search row too far below the clock */
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.75rem; }

.sug { padding:20px; border-radius:12px; font-family:Atkinson Hyperlegible Next,sans-serif; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); box-sizing: border-box; }
.sg { background:#ecfdf5 !important; border:1px solid #a7f3d0; color:#065f46 !important; }
.so { background:#fffbeb !important; border:1px solid #fde68a; color:#92400e !important; }
.sr { background:#fef2f2 !important; border:1px solid #fecaca; color:#991b1b !important; }

.stl { font-family:'Fraunces', Georgia, serif; font-optical-sizing:auto; font-size:1.45rem; font-weight:600; color:#1c1917 !important; margin:34px 0 10px 0; letter-spacing:-0.01em; display:flex; align-items:center; gap:8px; }
.sln { width:100%; height:1px; background:#e7e2d9; position:relative; margin:0 0 22px 0; }
.sln::before { content:''; position:absolute; left:0; top:-1px; width:34px; height:3px; background:#7a1420; border-radius:2px; }
.pk { font-family:Atkinson Hyperlegible Next,sans-serif; font-size:0.85rem; color:#57534e !important; font-weight: 600; margin-top: 0; background: #fffdf7 !important; padding: 8px 12px; border-radius: 8px; border: 1px solid #e3dcc8; box-sizing: border-box; }

.find-spot-title { font-family:'Fraunces', Georgia, serif; font-optical-sizing:auto; font-size:1.4rem; font-weight:600; color:#1c1917; margin:-5px 0 -4px 0; letter-spacing:-0.01em; line-height:1; }
.find-spot-sub { font-family:Atkinson Hyperlegible Next,sans-serif; font-size:0.85rem; font-weight:600; color:#0f172a; opacity:0.7; margin-bottom:8px; margin-top:2px; }

.detailed-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
.mini-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 16px; }

/* Zone cards */
.zc, .zc-mini { 
    background:#fffdf9 !important; 
    border:1px solid #e7e2d9; 
    box-shadow: 0 1px 2px rgba(28,25,23,0.04); 
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s ease; 
    cursor: pointer; 
    box-sizing: border-box; 
    width: 100%; 
    position: relative;
}
.zc { border-radius:14px; padding:20px; }
.zc-mini { border-radius:10px; padding:12px 12px 13px 12px; overflow: hidden; }

/* compact card layout: name + count on one line then one big colored
   bar. the bar track is tinted with the status color so even an empty
   zone still shows green at a glance */
.mn-row { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:8px; }
.mn-name { font-size:0.74rem; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; padding-top:2px; }
.mn-stat { display:flex; flex-direction:column; align-items:flex-end; gap:2px; flex-shrink:0; }
.mn-count { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:800; line-height:1; flex-shrink:0; }
.mn-cap { font-size:0.65rem; font-weight:600; }
.mn-bar { height:16px; border-radius:8px; overflow:hidden; box-shadow: inset 0 0 0 1px rgba(28,25,23,0.06); }
.zc-mini .mn-unit { font-family:'JetBrains Mono',monospace; font-size:0.48rem; font-weight:700; letter-spacing:0.9px; text-transform:uppercase; color:#a39a87 !important; white-space:nowrap; line-height:1; }
.mn-foot { display:flex; justify-content:flex-end; margin-top:6px; height:10px; }
.zc-mini .mn-cta { font-family:'JetBrains Mono',monospace; font-size:0.5rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#8a8273 !important; white-space:nowrap; line-height:1; opacity:0; transition:opacity 0.2s ease; }
.zc-mini:hover .mn-cta { opacity:1; }
@media (max-width: 640px) {
    .mn-foot { display:none; }
}
.mn-fill { height:100%; border-radius:8px; transition:width 0.6s cubic-bezier(0.4,0,0.2,1); box-shadow: inset 0 -4px 6px rgba(0,0,0,0.12); }
.zc-mini .mn-strip { height:20px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-size:0.56rem; font-weight:800; letter-spacing:0.6px; text-transform:uppercase; color:#ffffff !important; box-shadow: inset 0 0 0 1.5px rgba(255,255,255,0.3); padding:0 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

.zc:hover, .zc-mini:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 12px 22px -8px rgba(28,25,23,0.14); 
    border-color:#d6cfc3; 
}

.zc.selected, .zc-mini.selected {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -6px rgba(69, 10, 10, 0.25);
}

/* glow effect that follows the mouse over a card. the JS updates two
   css variables (one listener for the whole page, throttled). does not
   activate on touch screens since there is no hover */
.zc::after, .zc-mini::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: radial-gradient(170px circle at var(--mx, 50%) var(--my, 50%),
                rgba(122, 20, 32, 0.10), transparent 65%);
    opacity: 0;
    transition: opacity 0.25s ease;
    pointer-events: none;
}
.zc:hover::after, .zc-mini:hover::after { opacity: 1; }
html[data-theme="dark"] .zc::after, html[data-theme="dark"] .zc-mini::after {
    background: radial-gradient(170px circle at var(--mx, 50%) var(--my, 50%),
                rgba(252, 165, 165, 0.13), transparent 65%);
}

.zc *, .zc-mini * { pointer-events: none; color: #0f172a !important; }
.zr { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 12px; }
.zn { font-family:Atkinson Hyperlegible Next,sans-serif; font-weight:700; font-size:0.95rem; color:#0f172a !important; display:flex; align-items:center; gap:6px; }
.zp { padding:4px 11px; border-radius:5px; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.62rem; text-transform: uppercase; letter-spacing: 1px; color:#ffffff !important; box-shadow: inset 0 0 0 1.5px rgba(255,255,255,0.35); }
.zb { font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:800; margin:8px 0 4px 0; line-height: 1; }
.zm { font-family:Atkinson Hyperlegible Next,sans-serif; font-size:0.8rem; color:#78716c !important; margin-bottom: 12px; font-weight: 500; }
.bar { height:8px; background:#efe9dc; border-radius:4px; overflow:hidden; margin-top:12px; }
.fil { height:100%; border-radius:4px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }

.pg { background:#ecfdf5 !important; color:#059669 !important; }
.po { background:#fffbeb !important; color:#d97706 !important; }
.pr { background:#fef2f2 !important; color:#dc2626 !important; }
/* On zone-card pills the status color is the fill, not the tint. */
.zp.pg { background:#059669 !important; color:#ffffff !important; }
.zp.po { background:#d97706 !important; color:#ffffff !important; }
.zp.pr { background:#dc2626 !important; color:#ffffff !important; }

/* Booking pill on detailed cards */
.bk { margin-top:14px; padding:10px 12px; border-radius:8px; font-family:Atkinson Hyperlegible Next,sans-serif; font-size:0.78rem; font-weight:600; display:flex; align-items:center; gap:8px; line-height:1.4; }
.bk-wrap { display:flex; flex-direction:column; gap:2px; }
.bk-label { font-size:0.7rem; opacity:0.75; text-transform:uppercase; letter-spacing:0.5px; font-weight:700; }
.bk-val { font-size:0.85rem; font-weight:700; }
.br2 { background:#fef2f2 !important; color:#991b1b !important; border:1px solid #fecaca; }
.bg2 { background:#ecfdf5 !important; color:#065f46 !important; border:1px solid #a7f3d0; }
.bo2 { background:#fffbeb !important; color:#92400e !important; border:1px solid #fde68a; }

/* Booking pill inner rows (the "8:00 PM – 9:30 PM   BSIT 3-1" sub-rows
   inside the Discussion Room booking pill). Class-based instead of
   inline so dark-mode rules can override reliably. */
.bk-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 4px 6px;
    margin-top: 3px;
    background: rgba(255,255,255,0.5);
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}
.bk-row-name { opacity: 0.75; }

/* Mini-card booking badge (compact view of Discussion Room). Three
   variants: booked / scheduled-later / open. Class-based so dark-mode
   rules can override reliably without inline-style substring match. */
.mini-badge {
    margin-top: 8px;
    padding: 4px 6px;
    border-radius: 4px;
    font-family: Atkinson Hyperlegible Next, sans-serif;
    font-size: 0.55rem;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.mini-booked    { background: #fef2f2; color: #991b1b; }
.mini-scheduled { background: #fffbeb; color: #92400e; }
.mini-open      { background: #ecfdf5; color: #065f46; }

.lt { background:#ef4444 !important; color:#fff !important; padding:2px 6px; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.6rem; font-weight:800; margin-left:8px; animation:pulse 2s infinite; letter-spacing:0.5px;}

.cs { text-align:center; padding:48px 20px; background:rgba(122, 20, 32, 0.03) !important; border:2px dashed #d8cfb9; border-radius:16px; margin:16px 0 24px 0; }
.cs-title { font-family:'Fraunces', Georgia, serif; font-optical-sizing:auto; font-weight:600; font-size:1.3rem; color:#1c1917; margin:0; }
.cs-sub { font-family: Atkinson Hyperlegible Next, sans-serif; font-size: 0.95rem; color: #78716c; margin-top: 8px; }
html[data-theme="dark"] .cs-title { color: #f1f5f9 !important; }
html[data-theme="dark"] .cs-sub { color: #cbd5e1 !important; }
.ftr { text-align:center; font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.4px; color:#8a8273 !important; margin-top:48px; padding:24px; border-top:1px solid #e3dcc8; line-height: 1.9; }

[data-testid="stCheckbox"] { display: flex !important; justify-content: flex-end !important; }
[data-testid="stCheckbox"] > label p, [data-testid="stCheckbox"] > label span {
    color: #0f172a !important;
    font-family: Atkinson Hyperlegible Next, sans-serif !important;
    font-weight: 400 !important;
}

/* hide the "Press Enter to apply" hint that streamlit puts inside text
   inputs. it overlaps the password eye icon and looks broken on dark */
[data-testid="InputInstructions"] { display: none !important; }

/* chrome autofill puts a light background on dark inputs, this inset
   shadow trick fixes it */
html[data-theme="dark"] input:-webkit-autofill,
html[data-theme="dark"] input:-webkit-autofill:hover,
html[data-theme="dark"] input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 1000px #0f172a inset !important;
    -webkit-text-fill-color: #f1f5f9 !important;
    caret-color: #f1f5f9 !important;
}

/* the number input ("How many people?") had a black border from the
   default theme and blended into the background. style it like a card
   with a red focus ring */
[data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #ffffff !important;
    border: 1.5px solid #d3cab4 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: #7a1420 !important;
    box-shadow: 0 0 0 3px rgba(122, 20, 32, 0.12) !important;
}
[data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stNumberInput"] input {
    background: transparent !important;
    color: #0f172a !important;
    font-weight: 600;
}
[data-testid="stNumberInput"] button {
    background: transparent !important;
    border: none !important;
    color: #57534e !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #f3eee1 !important;
    color: #0f172a !important;
}

[data-testid="stPopover"] button {
    background: transparent !important;
    color: #6b6457 !important;
    border: 1px solid #ddd5c2 !important;
    border-radius: 8px !important;
    padding: 5px 10px !important;
    font-family: Atkinson Hyperlegible Next, sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    transition: all 0.15s ease !important;
    opacity: 0.7;
}
[data-testid="stPopover"] button:hover {
    background: #f3eee1 !important;
    border-color: #c9c0a9 !important;
    color: #44403c !important;
    opacity: 1;
}
[data-testid="stPopover"] button p,
[data-testid="stPopover"] button span {
    color: inherit !important;
}

/* The Streamlit popover trigger is a hidden anchor: invisible (opacity:0)
   and non-interactive (pointer-events:none), but positioned right where
   the visible header proxy button sits. Streamlit's popover panel
   anchors to this trigger, so the panel drops down directly under the
   header three-dot button. The proxy button calls .click() on the
   trigger to open it. */
/* The Streamlit popover trigger lives off-screen - its position must
   stay set so the popover panel anchors to a stable spot, but it's not
   clicked directly. The visible 3-dots button is a proxy in the header
   slot (created by JS) which forwards clicks to the trigger. */
.st-key-hidden_staff_trigger {
    position: fixed !important;
    top: 14px !important;
    right: 32px !important;
    width: 38px !important;
    height: 38px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    z-index: 999998 !important;
}
.st-key-hidden_staff_trigger > div,
.st-key-hidden_staff_trigger [data-testid="element-container"] {
    margin: 0 !important;
    padding: 0 !important;
}
#staff-proxy-btn {
    background: rgba(255,255,255,0.10);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 50%;
    width: 38px;
    height: 38px;
    padding: 0;
    font-family: Atkinson Hyperlegible Next, sans-serif;
    font-size: 1.1rem;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
}
#staff-proxy-btn:hover {
    background: rgba(255,255,255,0.20);
    border-color: rgba(255,255,255,0.35);
    transform: scale(1.05);
}
#staff-proxy-btn:focus { outline: none; box-shadow: 0 0 0 2px rgba(255,255,255,0.35); }

/* Theme toggle button - same circular icon style as the staff button */
#theme-toggle-btn {
    background: rgba(255,255,255,0.10);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 50%;
    width: 38px;
    height: 38px;
    padding: 0;
    font-family: Atkinson Hyperlegible Next, sans-serif;
    font-size: 1rem;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
}
#theme-toggle-btn:hover {
    background: rgba(255,255,255,0.20);
    border-color: rgba(255,255,255,0.35);
    transform: scale(1.05);
}
#theme-toggle-btn:focus { outline: none; box-shadow: 0 0 0 2px rgba(255,255,255,0.35); }

/* --- Map zoom controls (overlay on the floor plan only) --- */
.map-zoom-ctrl {
    position: absolute;
    right: 12px;
    bottom: 12px;
    z-index: 6;
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.map-zoom-ctrl button {
    width: 36px; height: 36px;
    border-radius: 10px;
    border: 1px solid rgba(28,25,23,0.12);
    background: rgba(255,255,255,0.94);
    color: #7a1420;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem; font-weight: 700; line-height: 1;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 8px rgba(28,25,23,0.16);
    transition: background 0.12s ease, transform 0.12s ease;
    -webkit-tap-highlight-color: transparent;
}
.map-zoom-ctrl button:hover { background: #fff; transform: scale(1.06); }
.map-zoom-ctrl button:active { transform: scale(0.94); }
.map-zoom-ctrl button[data-z="reset"] { font-size: 0.95rem; }
.map-zoom-hint {
    position: absolute;
    left: 12px; bottom: 12px; z-index: 6;
    font-family: Atkinson Hyperlegible Next, sans-serif; font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase;
    color: #7a1420;
    background: rgba(255,255,255,0.85);
    padding: 4px 9px; border-radius: 7px;
    border: 1px solid rgba(28,25,23,0.10);
    pointer-events: none;
    opacity: 0.85;
    transition: opacity 0.4s ease;
}
html[data-theme="dark"] .map-zoom-ctrl button {
    background: rgba(30,41,59,0.92);
    color: #fca5a5;
    border-color: rgba(255,255,255,0.14);
}
html[data-theme="dark"] .map-zoom-ctrl button:hover { background: #1e293b; }
html[data-theme="dark"] .map-zoom-hint {
    background: rgba(30,41,59,0.85); color: #fca5a5; border-color: rgba(255,255,255,0.12);
}
html[data-theme="dark"] .sln { background: #334155; }
html[data-theme="dark"] .sln::before { background: #fca5a5; }

/* --- Dark mode --- */
/* Toggled by setting data-theme="dark" on <html> (the JS does this).
   Using the html ancestor selector means every descendant inherits the
   theme regardless of how Streamlit nests its wrappers. */
html[data-theme="dark"],
html[data-theme="dark"] body,
html[data-theme="dark"] .stApp,
html[data-theme="dark"] [data-testid="stAppViewContainer"],
html[data-theme="dark"] [data-testid="stMain"],
html[data-theme="dark"] section.main {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}
html[data-theme="dark"] .clock { color: #fca5a5 !important; }
html[data-theme="dark"] [data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #1e293b !important;
    border-color: #334155 !important;
}
html[data-theme="dark"] [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: #fca5a5 !important;
    box-shadow: 0 0 0 3px rgba(252, 165, 165, 0.15) !important;
}
html[data-theme="dark"] [data-testid="stNumberInput"] input { color: #f1f5f9 !important; }
html[data-theme="dark"] [data-testid="stNumberInput"] button { color: #94a3b8 !important; }
html[data-theme="dark"] [data-testid="stNumberInput"] button:hover {
    background: #0f172a !important;
    color: #f1f5f9 !important;
}
html[data-theme="dark"] .stl { color: #f1f5f9 !important; }
html[data-theme="dark"] .pk { background: #1e293b !important; color: #cbd5e1 !important; border-color: #334155 !important; }

/* cards in dark mode */
html[data-theme="dark"] .zc, html[data-theme="dark"] .zc-mini {
    background: #1e293b !important;
    border-color: #334155 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
html[data-theme="dark"] .zc *, html[data-theme="dark"] .zc-mini * { color: #f1f5f9 !important; }
html[data-theme="dark"] .zc:hover, html[data-theme="dark"] .zc-mini:hover {
    box-shadow: 0 10px 20px -5px rgba(0,0,0,0.4);
}
html[data-theme="dark"] .bar { background: #0f172a !important; }
html[data-theme="dark"] .zm { color: #cbd5e1 !important; }
html[data-theme="dark"] .zc-mini .mn-unit { color: #64748b !important; }
html[data-theme="dark"] .zc-mini .mn-cta { color: #94a3b8 !important; }
html[data-theme="dark"] .zn { color: #ffffff !important; }

/* Status pills on cards - same bright treatment as the suggestion box
   so they pop against the dark card surface. Light text on saturated bg. */
html[data-theme="dark"] .zp.pg { background: #059669 !important; color: #ffffff !important; }
html[data-theme="dark"] .zp.po { background: #d97706 !important; color: #ffffff !important; }
html[data-theme="dark"] .zp.pr { background: #dc2626 !important; color: #ffffff !important; }

/* Mini-card status pill (different inline class - handled the same way) */
html[data-theme="dark"] .zc-mini [style*="background:#ecfdf5"] { background: #064e3b !important; color: #d1fae5 !important; }
html[data-theme="dark"] .zc-mini [style*="background:#fffbeb"] { background: #78350f !important; color: #fde68a !important; }
html[data-theme="dark"] .zc-mini [style*="background:#fef2f2"] { background: #7f1d1d !important; color: #fecaca !important; }

/* Class-based mini-badge dark variants - these reliably override
   regardless of how the browser normalizes the original inline style. */
html[data-theme="dark"] .mini-booked    { background: #7f1d1d !important; color: #fecaca !important; }
html[data-theme="dark"] .mini-scheduled { background: #78350f !important; color: #fde68a !important; }
html[data-theme="dark"] .mini-open      { background: #064e3b !important; color: #d1fae5 !important; }

/* Suggestion box keeps its colored variants but darker */
html[data-theme="dark"] .sug { box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3) !important; }
html[data-theme="dark"] .sug.sg { background: #064e3b !important; border-color: #047857 !important; color: #d1fae5 !important; }
html[data-theme="dark"] .sug.so { background: #78350f !important; border-color: #b45309 !important; color: #fde68a !important; }
html[data-theme="dark"] .sug.sr { background: #7f1d1d !important; border-color: #b91c1c !important; color: #fecaca !important; }
html[data-theme="dark"] .sug * { color: inherit !important; }

/* Popover panel (Library Personnel Access) - Streamlit uses a portal
   for the panel; wide selectors catch all variants of its container. */
html[data-theme="dark"] [data-testid="stPopover"],
html[data-theme="dark"] [data-baseweb="popover"],
html[data-theme="dark"] [data-baseweb="popover"] > div,
html[data-theme="dark"] [data-baseweb="popover"] [role="dialog"] {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #334155 !important;
}
html[data-theme="dark"] [data-baseweb="popover"] *,
html[data-theme="dark"] [role="dialog"] * { color: #f1f5f9 !important; }
html[data-theme="dark"] [data-baseweb="popover"] input,
html[data-theme="dark"] [data-baseweb="popover"] textarea,
html[data-theme="dark"] [role="dialog"] input,
html[data-theme="dark"] [role="dialog"] textarea {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}

/* Password input wrapper + eye-toggle button. Streamlit's default
   styling shows a white background to the right of the password field
   in dark mode - recolor everything inside the input wrapper to match
   the dark theme. */
html[data-theme="dark"] [data-baseweb="popover"] [data-baseweb="input"],
html[data-theme="dark"] [role="dialog"] [data-baseweb="input"],
html[data-theme="dark"] [data-baseweb="popover"] [data-baseweb="base-input"],
html[data-theme="dark"] [role="dialog"] [data-baseweb="base-input"] {
    background: #0f172a !important;
    border-color: #475569 !important;
}
html[data-theme="dark"] [data-baseweb="popover"] [data-baseweb="input"] button,
html[data-theme="dark"] [role="dialog"] [data-baseweb="input"] button {
    background: transparent !important;
    color: #cbd5e1 !important;
}
html[data-theme="dark"] [data-baseweb="popover"] [data-baseweb="input"] svg,
html[data-theme="dark"] [role="dialog"] [data-baseweb="input"] svg {
    fill: #cbd5e1 !important;
    color: #cbd5e1 !important;
}

/* Inline `<code>` (markdown backticks like `9:00 AM`) inside the
   popover panel - Streamlit's default light gray background looks
   like an unreadable white pill on the dark theme. */
html[data-theme="dark"] [data-baseweb="popover"] code,
html[data-theme="dark"] [role="dialog"] code {
    background: rgba(255,255,255,0.10) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Buttons inside the popover (delete ✕, form submit, etc.) need a
   dark fill; the global .stButton dark rule isn't always specific
   enough to win against Streamlit's portal-rendered defaults. */
html[data-theme="dark"] [data-baseweb="popover"] button,
html[data-theme="dark"] [role="dialog"] button {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}
html[data-theme="dark"] [data-baseweb="popover"] button:hover,
html[data-theme="dark"] [role="dialog"] button:hover {
    background: #334155 !important;
    border-color: #64748b !important;
}
html[data-theme="dark"] [data-baseweb="popover"] [kind="primary"],
html[data-theme="dark"] [data-baseweb="popover"] [kind="primaryFormSubmit"],
html[data-theme="dark"] [role="dialog"] [kind="primary"],
html[data-theme="dark"] [role="dialog"] [kind="primaryFormSubmit"] {
    background: #7f1d1d !important;
    border-color: #7f1d1d !important;
    color: #ffffff !important;
}

/* Date/time picker inputs - Streamlit wraps these in containers with
   white backgrounds and embedded calendar/clock icon buttons. */
html[data-theme="dark"] [data-baseweb="popover"] [data-testid="stDateInput"] *,
html[data-theme="dark"] [data-baseweb="popover"] [data-testid="stTimeInput"] *,
html[data-theme="dark"] [data-baseweb="popover"] [data-baseweb="select"] *,
html[data-theme="dark"] [role="dialog"] [data-testid="stDateInput"] *,
html[data-theme="dark"] [role="dialog"] [data-testid="stTimeInput"] *,
html[data-theme="dark"] [role="dialog"] [data-baseweb="select"] * {
    background-color: transparent !important;
}
html[data-theme="dark"] [data-baseweb="popover"] [data-testid="stDateInput"] > div,
html[data-theme="dark"] [data-baseweb="popover"] [data-testid="stTimeInput"] > div,
html[data-theme="dark"] [data-baseweb="popover"] [data-baseweb="select"] > div,
html[data-theme="dark"] [role="dialog"] [data-testid="stDateInput"] > div,
html[data-theme="dark"] [role="dialog"] [data-testid="stTimeInput"] > div,
html[data-theme="dark"] [role="dialog"] [data-baseweb="select"] > div {
    background: #0f172a !important;
    border-color: #475569 !important;
}

/* The calendar DROPDOWN itself opens in a separate baseweb portal
   ([data-baseweb="calendar"]), NOT inside the popover/dialog - so the
   rules above never reach it and it rendered with broken white-on-white
   colors in dark mode. These rules style that dropdown directly: the
   calendar surface, month/weekday labels, day cells, hover, and the
   selected day. Scoped to dark theme only. */
html[data-theme="dark"] [data-baseweb="calendar"],
html[data-theme="dark"] [data-baseweb="calendar"] > div,
html[data-theme="dark"] [data-baseweb="calendar"] [data-baseweb="calendar-month"] {
    background: #0f172a !important;
    color: #f1f5f9 !important;
}
html[data-theme="dark"] [data-baseweb="calendar"] * {
    color: #f1f5f9 !important;
}
/* Month/year header and the prev/next arrows */
html[data-theme="dark"] [data-baseweb="calendar"] button {
    background: transparent !important;
    color: #f1f5f9 !important;
    border: none !important;
}
html[data-theme="dark"] [data-baseweb="calendar"] button:hover {
    background: #1e293b !important;
}
/* Individual day cells (role="gridcell") */
html[data-theme="dark"] [data-baseweb="calendar"] [role="gridcell"] > div {
    background: transparent !important;
    color: #f1f5f9 !important;
}
html[data-theme="dark"] [data-baseweb="calendar"] [role="gridcell"] > div:hover {
    background: #334155 !important;
    color: #ffffff !important;
}
/* The selected day keeps the brand red so it stays legible */
html[data-theme="dark"] [data-baseweb="calendar"] [aria-selected="true"] > div,
html[data-theme="dark"] [data-baseweb="calendar"] [aria-selected="true"] {
    background: #7f1d1d !important;
    color: #ffffff !important;
}
/* Days outside the current month / disabled days: dim them */
html[data-theme="dark"] [data-baseweb="calendar"] [aria-disabled="true"] > div {
    color: #475569 !important;
}

/* Inline-styled text in markdown blocks (Find a spot title, How many
   people subtitle, suggestion box content, Floor 2 placeholder) sets
   color:#0f172a directly. These overrides recolor them for dark mode
   without needing to rewrite the inline styles everywhere. */
html[data-theme="dark"] [style*="color:#0f172a"],
html[data-theme="dark"] [style*="color: #0f172a"] {
    color: #f1f5f9 !important;
}
html[data-theme="dark"] [style*="background:rgba(0,0,0,0.05)"] {
    background: rgba(255,255,255,0.12) !important;
}

/* Floor map: keep the cream OUTER background as a wayfinding-sheet look,
   but recolor the INDIVIDUAL ROOM rectangles to dark slate so they
   contrast against the cream and the legend text reads cleanly. */
html[data-theme="dark"] svg.map-interactive {
    background: #555d50 !important;
}
html[data-theme="dark"] svg.map-interactive > rect:first-child {
    fill: #555d50 !important;
}
html[data-theme="dark"] svg.map-interactive .w {
    fill: #1e293b !important;
    stroke: #475569 !important;
}
html[data-theme="dark"] svg.map-interactive .n { fill: #f1f5f9 !important; }
html[data-theme="dark"] svg.map-interactive .g { fill: #cbd5e1 !important; }
html[data-theme="dark"] svg.map-interactive > rect:first-child { stroke: #3a4252 !important; }
html[data-theme="dark"] svg.map-interactive .map-static rect.w { fill: #182234 !important; stroke: #2c3a52 !important; }
html[data-theme="dark"] svg.map-interactive .map-static .n { fill: #94a3b8 !important; }
html[data-theme="dark"] svg.map-interactive .t { fill: #94a3b8 !important; }
html[data-theme="dark"] svg.map-interactive .p { fill: #6e7565 !important; }
html[data-theme="dark"] svg.map-interactive line { stroke: #475569 !important; }
html[data-theme="dark"] svg.map-interactive text[fill="#64748b"] { fill: #94a3b8 !important; }
html[data-theme="dark"] svg.map-interactive .ic { fill: #8fa0b8 !important; }
html[data-theme="dark"] svg.map-interactive rect[fill="#54422a"] { fill: #7a6240 !important; }

/* Plotly chart text */
html[data-theme="dark"] .js-plotly-plot text { fill: #cbd5e1 !important; }

/* Footer */
html[data-theme="dark"] .ftr { color: #94a3b8 !important; border-top-color: #334155 !important; }

/* Streamlit buttons in dark mode */
html[data-theme="dark"] .stButton > button {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}
html[data-theme="dark"] .stButton > button:hover {
    background: #334155 !important;
    border-color: #64748b !important;
}
html[data-theme="dark"] .stButton > button[kind="primary"] {
    background: #7f1d1d !important;
    color: #ffffff !important;
    border-color: #7f1d1d !important;
}

/* Admin tabs and inputs */
html[data-theme="dark"] [data-baseweb="tab"] { color: #cbd5e1 !important; }
html[data-theme="dark"] .stTabs [data-baseweb="tab-list"] { background: #1e293b !important; }
/* Toggle in dark mode: text and switch knob/track recolored.
   opacity:1 prevents BaseWeb's faded unchecked-label styling
   from making white text look gray. */
html[data-theme="dark"] [data-testid="stCheckbox"] *:not([role="checkbox"]):not([role="checkbox"] *) {
    color: #ffffff !important;
    opacity: 1 !important;
}
/* Also explicitly target the widget label container - emotion sets the
   color inline via "Label.color: bodyText" so we need a high-specificity
   override that wins regardless of class-name ordering. */
html[data-theme="dark"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
html[data-theme="dark"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] *,
html[data-theme="dark"] [data-testid="stCheckbox"] label > div,
html[data-theme="dark"] [data-testid="stCheckbox"] label > div * {
    color: #ffffff !important;
    opacity: 1 !important;
}
html[data-theme="dark"] [data-testid="stCheckbox"] [role="checkbox"] {
    background: #475569 !important;
}
html[data-theme="dark"] [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
    background: #ef4444 !important;
}
html[data-theme="dark"] [data-testid="stCheckbox"] [role="checkbox"] > div,
html[data-theme="dark"] [data-testid="stCheckbox"] [role="checkbox"] > span {
    background: #ffffff !important;
}

/* Find-a-spot title + subtitle (now using CSS classes) */
html[data-theme="dark"] .find-spot-title { color: #f1f5f9 !important; }
html[data-theme="dark"] .find-spot-sub { color: #cbd5e1 !important; }

/* Booking pills inside cards - keep their colored identity (red/orange/
   green) but use brighter saturated dark variants so the body text is
   readable against a darker tinted background. */
html[data-theme="dark"] .bk.br2 { background: #7f1d1d !important; color: #fecaca !important; border-color: #b91c1c !important; }
html[data-theme="dark"] .bk.bg2 { background: #064e3b !important; color: #d1fae5 !important; border-color: #047857 !important; }
html[data-theme="dark"] .bk.bo2 { background: #78350f !important; color: #fde68a !important; border-color: #b45309 !important; }
html[data-theme="dark"] .bk * { color: inherit !important; }

/* Booking pill inner rows in dark mode - translucent dark instead
   of translucent white so text stays readable on saturated backgrounds. */
html[data-theme="dark"] .bk-row {
    background: rgba(0,0,0,0.35) !important;
    color: inherit !important;
}
html[data-theme="dark"] .bk-row * { color: inherit !important; }
/* Legacy inline-style fallback */
html[data-theme="dark"] [style*="background:rgba(255,255,255,0.5)"] {
    background: rgba(0,0,0,0.35) !important;
}

/* Mini-card booking badges (Discussion Room compact view) - also
   inline-styled, override the pale backgrounds with their dark variants.
   The fef2f2 variant lives elsewhere in this stylesheet. */
html[data-theme="dark"] .zc-mini [style*="background:#fffbeb"]:not(span) { background: #78350f !important; color: #fde68a !important; }
html[data-theme="dark"] .zc-mini [style*="background:#ecfdf5"]:not(span) { background: #064e3b !important; color: #d1fae5 !important; }

/* Admin booking banners ("Currently Occupied" / "Available now") use
   inline pale backgrounds with dark text - flip both sides for dark mode */
html[data-theme="dark"] [data-baseweb="popover"] [style*="background:#fef2f2"],
html[data-theme="dark"] [role="dialog"] [style*="background:#fef2f2"] {
    background: #7f1d1d !important;
    border-color: #b91c1c !important;
}
html[data-theme="dark"] [data-baseweb="popover"] [style*="background:#ecfdf5"],
html[data-theme="dark"] [role="dialog"] [style*="background:#ecfdf5"] {
    background: #064e3b !important;
    border-color: #047857 !important;
}
html[data-theme="dark"] [data-baseweb="popover"] [style*="background:#fffbeb"],
html[data-theme="dark"] [role="dialog"] [style*="background:#fffbeb"] {
    background: #78350f !important;
    border-color: #b45309 !important;
}

/* Admin "Access granted" success - Streamlit's stAlert with success kind */
html[data-theme="dark"] [data-testid="stAlertContainer"][kind="success"],
html[data-theme="dark"] [data-baseweb="notification"][kind="success"] {
    background: #064e3b !important;
}

/* Inline-styled text in markdown that hardcodes #991b1b / #065f46 / #92400e
   (admin banner inner text) - keep readable on dark surfaces */
html[data-theme="dark"] [style*="color:#991b1b"] { color: #fecaca !important; }
html[data-theme="dark"] [style*="color:#065f46"] { color: #d1fae5 !important; }
html[data-theme="dark"] [style*="color:#92400e"] { color: #fde68a !important; }

/* === Class-based admin banners === */
.status-banner {
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 10px;
    border: 1px solid transparent;
}
.status-banner .banner-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.status-banner .banner-body {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 2px;
}
.banner-occupied { background: #fef2f2; border-color: #fecaca; }
.banner-occupied .banner-label { color: #991b1b; }
.banner-occupied .banner-body  { color: #0f172a; }
.banner-available { background: #ecfdf5; border-color: #a7f3d0; }
.banner-available .banner-label { color: #065f46; }
.banner-available .banner-body  { color: #0f172a; }

html[data-theme="dark"] .banner-occupied { background: #7f1d1d !important; border-color: #b91c1c !important; }
html[data-theme="dark"] .banner-occupied .banner-label { color: #fecaca !important; }
html[data-theme="dark"] .banner-occupied .banner-body  { color: #ffffff !important; }
html[data-theme="dark"] .banner-available { background: #064e3b !important; border-color: #047857 !important; }
html[data-theme="dark"] .banner-available .banner-label { color: #d1fae5 !important; }
html[data-theme="dark"] .banner-available .banner-body  { color: #ffffff !important; }

/* === Class-based status pills (camera worker / system) === */
.status-pill {
    padding: 8px 12px;
    border-radius: 8px;
    font-family: Atkinson Hyperlegible Next, sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
}
html[data-theme="dark"] .status-pill-success { background: #064e3b !important; color: #d1fae5 !important; border-color: #047857 !important; }
html[data-theme="dark"] .status-pill-warning { background: #78350f !important; color: #fde68a !important; border-color: #b45309 !important; }
html[data-theme="dark"] .status-pill-error   { background: #7f1d1d !important; color: #fecaca !important; border-color: #b91c1c !important; }
html[data-theme="dark"] .status-pill-neutral { background: #1e293b !important; color: #cbd5e1 !important; border-color: #475569 !important; }

/* === Streamlit divider, warnings, infos in dark mode === */
html[data-theme="dark"] hr,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] hr {
    border-color: #475569 !important;
    background-color: #475569 !important;
}
html[data-theme="dark"] [data-testid="stAlertContainer"] {
    border-color: #475569 !important;
}
html[data-theme="dark"] [data-testid="stAlertContainer"][kind="info"] {
    background: #1e3a5f !important;
}
html[data-theme="dark"] [data-testid="stAlertContainer"][kind="warning"] {
    background: #78350f !important;
}
html[data-theme="dark"] [data-testid="stAlertContainer"][kind="error"] {
    background: #7f1d1d !important;
}
html[data-theme="dark"] [data-testid="stAlertContainer"][kind="success"] {
    background: #064e3b !important;
}
html[data-theme="dark"] [data-testid="stAlertContainer"] * { color: #f1f5f9 !important; }

/* st.code blocks inside dialog/popover (the pip install hint, etc.) */
html[data-theme="dark"] [data-testid="stCode"],
html[data-theme="dark"] pre {
    background: #0f172a !important;
    border-color: #475569 !important;
}
html[data-theme="dark"] [data-testid="stCode"] *,
html[data-theme="dark"] pre * { color: #f1f5f9 !important; }

/* st.dialog body - same treatment as popover */
html[data-theme="dark"] [role="dialog"] {
    background: #1e293b !important;
}
html[data-theme="dark"] [role="dialog"] *:not(canvas):not(svg):not(img) { color: #f1f5f9; }

/* st.caption inside dialog/popover */
html[data-theme="dark"] [data-baseweb="popover"] [data-testid="stCaptionContainer"],
html[data-theme="dark"] [role="dialog"] [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* zone coordinate pill in the ROI editor (the "(x, y) - size WxH" line) */
.zone-coord {
    padding: 6px 10px;
    background: #efe9dc;
    border-radius: 4px;
    font-family: JetBrains Mono, monospace;
    font-size: 0.8rem;
    color: #0f172a;
}
html[data-theme="dark"] .zone-coord {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border: 1px solid #475569 !important;
}

/* Date picker CALENDAR in dark mode (baseweb). The day grid and the
   month-navigation arrows were rendering on a white surface, making the
   numbers hard to read. Force a dark surface + light text/arrows, while
   leaving the SELECTED day's accent color intact. */
html[data-theme="dark"] [data-baseweb="calendar"],
html[data-theme="dark"] [data-baseweb="calendar"] > div,
html[data-theme="dark"] [data-baseweb="datepicker"],
html[data-theme="dark"] [data-baseweb="calendar"] [role="grid"] {
    background: #1e293b !important;
}
/* All calendar text light by default (month label, weekday headers, days) */
html[data-theme="dark"] [data-baseweb="calendar"] * {
    color: #f1f5f9 !important;
}
/* Weekday header row (Mo Tu We...) a touch dimmer */
html[data-theme="dark"] [data-baseweb="calendar"] [role="columnheader"] {
    color: #94a3b8 !important;
}
/* Month-navigation arrows (prev/next) */
html[data-theme="dark"] [data-baseweb="calendar"] button {
    background: transparent !important;
}
html[data-theme="dark"] [data-baseweb="calendar"] button svg,
html[data-theme="dark"] [data-baseweb="calendar"] svg {
    fill: #f1f5f9 !important;
    color: #f1f5f9 !important;
}
/* Day cells: dark surface unless they are the selected day */
html[data-theme="dark"] [data-baseweb="calendar"] [role="gridcell"] > div:not([aria-selected="true"]) {
    background: #1e293b !important;
}
/* Hovered day for feedback */
html[data-theme="dark"] [data-baseweb="calendar"] [role="gridcell"] > div:not([aria-selected="true"]):hover {
    background: #334155 !important;
}

/* st.form border (Add new booking, etc.) in dark mode. */
html[data-theme="dark"] [data-testid="stForm"],
html[data-theme="dark"] form[data-testid="stForm"] {
    border-color: #475569 !important;
    background: transparent !important;
}

/* Drawable canvas component renders inside an iframe in the ROI dialog.
   Default white background bleeds where the canvas doesn't fill it. */
html[data-theme="dark"] [role="dialog"] iframe,
html[data-theme="dark"] [role="dialog"] [data-testid="stIFrame"],
html[data-theme="dark"] [role="dialog"] [data-testid^="stCustomComponent"] {
    background: #1e293b !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: #fffdf9 !important;
    color: #44403c !important;
    border: 1.5px solid #d3cab4 !important;
    border-radius: 10px !important;
    font-family: Atkinson Hyperlegible Next, sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 12px 24px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #f3eee1 !important;
    border-color: #b0a78e !important;
    color: #0f172a !important;
}

.stButton > button[kind="primary"] {
    background: #450a0a !important;
    color: #ffffff !important;
    border-color: #450a0a !important;
}
.stButton > button[kind="primary"]:hover {
    background: #220505 !important;
}

@media (max-width:768px) { 
    .block-container { padding-top: 0.75rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; } 
    .mini-grid { grid-template-columns: 1fr 1fr; gap: 8px; }
    .detailed-grid { grid-template-columns: 1fr; gap: 12px; }
    [data-testid="stPopover"] button {
        padding: 4px 8px !important;
        font-size: 0.7rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

counts = get_current_counts()
now = datetime.now()
today_str = now.strftime("%Y-%m-%d")
tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")


# --- Header ---
# Rendered via components.html so the header lives in its own iframe.
# An iframe with position:fixed inside it works against the iframe's own
# viewport, and we can position the iframe itself with fixed CSS targeting
# its parent. We send a height=0 iframe but the inner content escapes via
# position:fixed on document.body - same idea as a portal in React.
logo_html = ""
if LOGO_PATH.exists():
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="hdr-logo">'

components.html(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
      html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
      /* The pinned bar's horizontal position is set by JS to match
         .block-container's actual on-page position. CSS just provides
         the visual styling. Width and left are dynamic. */
      #pinned-wrap {{
          position: fixed;
          top: 0;
          z-index: 999999;
          pointer-events: none;
          box-sizing: border-box;
      }}
      #pinned-bar {{
          background: linear-gradient(135deg, #450a0a 0%, #220505 100%);
          padding: 6px 14px;
          border-radius: 0 0 14px 14px;
          display: flex;
          align-items: center;
          gap: 14px;
          border-bottom: 3px solid #b8902c;
          box-shadow: 0 6px 16px -4px rgba(0,0,0,0.25);
          box-sizing: border-box;
          pointer-events: auto;
          width: 100%;
      }}
      .hdr-logo {{ height: 60px; align-self: center; transform: translateY(2px); }}
      .hdr-t {{ font-family: 'Fraunces', Georgia, serif; font-optical-sizing: auto; font-weight: 600; font-size: 1.5rem; color: #fff; line-height: 1.05; margin: 0; letter-spacing: -0.01em; }}
      .hdr-s {{ font-family: Atkinson Hyperlegible Next, sans-serif; font-size: 0.62rem; color: #dcc389; letter-spacing: 2.5px; text-transform: uppercase; margin: 2px 0 0 0; font-weight: 600; }}
      @media (max-width: 768px) {{
          #pinned-bar {{ padding: 4px 8px; gap: 10px; }}
          .hdr-logo {{ height: 48px; display: block; object-fit: contain; }}
          .hdr-t {{ font-size: 1.2rem; }}
          .hdr-s {{ font-size: 0.52rem; letter-spacing: 2px; }}
      }}
    </style>
    <div id="pinned-wrap">
      <div id="pinned-bar">
          {logo_html}
          <div>
              <div class="hdr-t">LOC-LOC</div>
              <div class="hdr-s">Real-Time Seat Spot Finder</div>
          </div>
          <div id="pinned-staff-slot" style="margin-left:auto;display:flex;align-items:center;gap:8px;"></div>
      </div>
    </div>
    <script>
      // Move this header out of its iframe into the parent document so
      // position:fixed sticks to the actual page viewport, not the iframe.
      // The iframe itself collapses to zero height; only the moved bar
      // remains visible.
      (function() {{
        try {{
          const wrap = document.getElementById('pinned-wrap');
          const styleEl = document.querySelector('style');
          const pDoc = window.parent.document;
          const pWin = window.parent;
          if (!wrap || !pDoc) return;

          // Drop any old copy from a previous run
          const oldWrap = pDoc.getElementById('pinned-wrap');
          if (oldWrap) oldWrap.remove();
          const oldFill = pDoc.getElementById('pinned-topfill');
          if (oldFill) oldFill.remove();
          const oldStyle = pDoc.getElementById('loc-pinned-style');
          if (oldStyle) oldStyle.remove();

          // Move our style + wrapper into the parent document
          if (styleEl) {{
            const styleClone = pDoc.createElement('style');
            styleClone.id = 'loc-pinned-style';
            styleClone.textContent = styleEl.textContent;
            pDoc.head.appendChild(styleClone);
          }}
          pDoc.body.appendChild(wrap);

          // Reserve space at the top and bottom of the page using REAL
          // DOM spacer elements inside .block-container.
          const adjustSpacers = () => {{
            const bc = pDoc.querySelector('.block-container');
            if (!bc) return;
            const isMobile = pWin.innerWidth <= 768;
            // Top spacer: header is smaller on mobile (smaller logo + tighter
            // padding via the @media rule), so the spacer is smaller too.
            let topSpacer = pDoc.getElementById('loc-top-spacer');
            if (!topSpacer) {{
              topSpacer = pDoc.createElement('div');
              topSpacer.id = 'loc-top-spacer';
              topSpacer.style.flexShrink = '0';
              topSpacer.style.pointerEvents = 'none';
              bc.insertBefore(topSpacer, bc.firstChild);
            }}
            const bar = pDoc.getElementById('pinned-bar');
            const barH = bar ? bar.offsetHeight : (isMobile ? 56 : 72);
            // self correcting: measure where the clock actually landed
            // and adjust the spacer until it is 2px below the header bar
            const probe = pDoc.querySelector('.clock');
            if (probe) {{
                const cur = parseFloat(topSpacer.style.height) || 0;
                // streamlit scrolls inside its own container not the
                // window. add up all the scroll positions (the ones
                // that dont scroll just give 0)
                const sMain = pDoc.querySelector('[data-testid="stMain"]');
                const sApp = pDoc.querySelector('[data-testid="stAppViewContainer"]');
                const s = (pWin.scrollY || 0)
                    + (sMain ? sMain.scrollTop : 0)
                    + (sApp ? sApp.scrollTop : 0)
                    + (pDoc.documentElement ? pDoc.documentElement.scrollTop : 0);
                const probeTop = probe.getBoundingClientRect().top + s;
                const delta = (barH + 2) - probeTop;
                if (Math.abs(delta) > 1) {{
                    // limit the spacer so a bad measurement cant make
                    // the gap grow forever
                    const next = Math.min(Math.max(0, cur + delta), barH + 80);
                    topSpacer.style.height = next + 'px';
                }}
            }} else {{
                topSpacer.style.height = (barH + 2) + 'px';
            }}
            // Bottom spacer: small buffer below the footer - just enough
            // breathing room so the LPU Manila line isn't flush with the
            // viewport edge when scrolled to the end.
            let bottomSpacer = pDoc.getElementById('loc-bottom-spacer');
            if (!bottomSpacer) {{
              bottomSpacer = pDoc.createElement('div');
              bottomSpacer.id = 'loc-bottom-spacer';
              bottomSpacer.style.flexShrink = '0';
              bottomSpacer.style.pointerEvents = 'none';
              bc.appendChild(bottomSpacer);
            }}
            bottomSpacer.style.height = '8px';
            // Strip any stale padding the older code left on the app
            // container.
            const app = pDoc.querySelector('[data-testid="stAppViewContainer"]');
            if (app) {{
              app.style.paddingTop = '';
              app.style.paddingBottom = '';
            }}
          }};
          adjustSpacers();
          if (pWin._locPaddingHandler) {{
            pWin.removeEventListener('resize', pWin._locPaddingHandler);
          }}
          pWin._locPaddingHandler = adjustSpacers;
          pWin.addEventListener('resize', adjustSpacers);

          // Match header position to .block-container exactly. Reading
          // its computed left + content width on every layout change
          // keeps the header aligned regardless of viewport, scrollbar,
          // or sidebar state. Using the inner content box (without
          // padding) means the header bar sits flush with the cards
          // and map below.
          function alignHeader() {{
            const bc = pDoc.querySelector('.block-container');
            if (!bc) return;
            const rect = bc.getBoundingClientRect();
            const cs = pWin.getComputedStyle(bc);
            const padL = parseFloat(cs.paddingLeft) || 0;
            const padR = parseFloat(cs.paddingRight) || 0;
            const newLeft = Math.round(rect.left + padL);
            const newWidth = Math.round(rect.width - padL - padR);
            // Only WRITE when the value actually changed. Without this guard
            // the MutationObserver below (which fires on every fragment
            // refresh) rewrote left/width every tick with identical values,
            // and those style writes triggered more mutations - a feedback
            // loop that showed up as the header button flickering and
            // visibly jumping, especially during/after a window resize.
            if (wrap._locLeft !== newLeft) {{
              wrap.style.left = newLeft + 'px';
              wrap._locLeft = newLeft;
            }}
            if (wrap._locWidth !== newWidth) {{
              wrap.style.width = newWidth + 'px';
              wrap._locWidth = newWidth;
            }}
          }}
          alignHeader();

          // Re-align on viewport resize and whenever Streamlit's DOM
          // changes (which can shift the block-container).
          if (pWin._locAlignBound) {{
            pWin.removeEventListener('resize', pWin._locAlignBound);
          }}
          pWin._locAlignBound = alignHeader;
          pWin.addEventListener('resize', alignHeader);

          if (pDoc._locAlignObs) pDoc._locAlignObs.disconnect();
          pDoc._locAlignObs = new MutationObserver(() => {{
            // Throttle to one tick - multiple mutations per frame
            // shouldn't cause repeated layout reads. Also re-run the
            // spacer setup in case Streamlit rebuilt .block-container
            // (which would have wiped out our injected spacer divs).
            if (pWin._locAlignPending) return;
            pWin._locAlignPending = true;
            pWin.requestAnimationFrame(() => {{
              pWin._locAlignPending = false;
              alignHeader();
              adjustSpacers();
            }});
          }});
          // childList only: observing `attributes` made the observer fire on
          // the header's own style writes, feeding the flicker loop.
          pDoc._locAlignObs.observe(pDoc.body, {{ childList: true, subtree: true }});
        }} catch (e) {{
          console.error('Failed to pin header:', e);
        }}
      }})();
    </script>
    """,
    height=0,
)

st.markdown(
    f'<div class="clock">{now.strftime("%I:%M %p &mdash; %B %d, %Y")}</div>',
    unsafe_allow_html=True,
)

# Admin popover - rendered as a three-dot icon button. The JS bridge later
# physically moves it into the fixed header at the far right.
# --- Library Personnel Access content ---
# Rendered inside the staff popover after a correct password is entered.
# The popover lives in a hidden container; the JS bridge surfaces it
# under the header three-dot button.
def staff_panel_content():
    st.success("✓ Access granted")

    tab_bookings, tab_camera, tab_system, tab_password = st.tabs(
        ["📅 Bookings", "📹 Camera", "ℹ️ System", "🔑 Password"]
    )

    with tab_bookings:
        current = get_current_booking(st.session_state.bookings)
        if current:
            st.markdown(
                f'<div class="status-banner banner-occupied">'
                f'<div class="banner-label">Currently Occupied</div>'
                f'<div class="banner-body">{current["booked_by"]} · until {current["end_time"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-banner banner-available">'
                '<div class="banner-label">Available Now</div>'
                '<div class="banner-body">No active booking</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("**All scheduled bookings:**")

        if not st.session_state.bookings:
            st.caption("No bookings scheduled.")
        else:
            sorted_bks = sorted(
                st.session_state.bookings,
                key=lambda b: (b.get("date", today_str), b.get("start_time", "")),
            )
            delete_id = None
            for bk in sorted_bks:
                b_date = bk.get("date", today_str)
                if b_date == today_str:
                    dlabel = "Today"
                elif b_date == tomorrow_str:
                    dlabel = "Tomorrow"
                else:
                    try:
                        dlabel = datetime.strptime(b_date, "%Y-%m-%d").strftime("%b %d, %Y")
                    except ValueError:
                        dlabel = b_date

                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"**{dlabel}** · `{bk['start_time']}` – `{bk['end_time']}`  \n"
                        f"**{bk['booked_by']}** _{bk.get('purpose', '-')}_"
                    )
                with c2:
                    if st.button("✕", key=f"del_{bk['id']}", help="Delete"):
                        delete_id = bk["id"]

            if delete_id:
                st.session_state.bookings = [b for b in st.session_state.bookings if b["id"] != delete_id]
                st.session_state.booking_version += 1
                st.rerun()

        st.divider()
        st.markdown("**Add new booking:**")

        with st.form("new_booking", clear_on_submit=True):
            new_by = st.text_input("Group / Class name", placeholder="BSIT 3-2", key="bk_new_by")
            new_purpose = st.text_input("Purpose (optional)", placeholder="Group meeting", key="bk_new_purpose")
            new_date = st.date_input("Date", value=now.date(), min_value=now.date(), key="bk_new_date")

            st.caption("Start time")
            s1, s2, s3 = st.columns(3)
            with s1:
                sh = st.selectbox("H", list(range(1, 13)), index=8, label_visibility="collapsed", key="bk_start_hour")
            with s2:
                sm = st.selectbox("M", [f"{m:02d}" for m in range(0, 60, 5)], index=0, label_visibility="collapsed", key="bk_start_min")
            with s3:
                sap = st.selectbox("AP", ["AM", "PM"], index=0, label_visibility="collapsed", key="bk_start_ampm")

            st.caption("End time")
            e1, e2, e3 = st.columns(3)
            with e1:
                eh = st.selectbox("H", list(range(1, 13)), index=10, label_visibility="collapsed", key="bk_end_hour")
            with e2:
                em = st.selectbox("M", [f"{m:02d}" for m in range(0, 60, 5)], index=0, label_visibility="collapsed", key="bk_end_min")
            with e3:
                eap = st.selectbox("AP", ["AM", "PM"], index=0, label_visibility="collapsed", key="bk_end_ampm")

            if st.form_submit_button("Add booking", type="primary", width='stretch'):
                if new_by:
                    st.session_state.booking_counter += 1
                    st.session_state.bookings.append({
                        "id": f"bk{st.session_state.booking_counter:03d}",
                        "date": new_date.strftime("%Y-%m-%d"),
                        "booked_by": new_by,
                        "purpose": new_purpose or "-",
                        "start_time": f"{sh}:{sm} {sap}",
                        "end_time": f"{eh}:{em} {eap}",
                    })
                    st.session_state.booking_version += 1
                    st.rerun()
                else:
                    st.error("Group / Class name is required.")

    with tab_camera:
        st.markdown("#### Live Camera Feed - Book Common Area 2")
        render_camera_status_fragment()

        static_snapshot = PROJECT_DIR / "static" / "latest_frame.jpg"
        try:
            static_snapshot.parent.mkdir(exist_ok=True)
            if SNAPSHOT_PATH.exists() and (
                not static_snapshot.exists()
                or SNAPSHOT_PATH.stat().st_mtime > static_snapshot.stat().st_mtime
            ):
                import shutil
                _tmp = static_snapshot.with_suffix(".jpg.tmp")
                shutil.copy2(SNAPSHOT_PATH, _tmp)
                os.replace(_tmp, static_snapshot)
        except OSError:
            pass

        feed_mtime = safe_mtime(static_snapshot)
        if feed_mtime is not None:
            initial_ts = int(feed_mtime)
            st.markdown(
                f'<img id="loc-live-feed" '
                f'src="./app/static/latest_frame.jpg?t={initial_ts}" '
                f'style="width:100%; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); display:block;" />'
                f'<div style="font-family:Atkinson Hyperlegible Next,sans-serif;font-size:0.85rem;color:#64748b;margin-top:6px;text-align:center;">'
                f'Live detection · Teal rectangles = Activity Zones · Green dots = counted · Yellow dots = ignored'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No camera feed available yet. Start the camera worker to see the live view.")

        st.caption(get_detection_label())

        st.divider()
        st.markdown("#### Activity Zones (ROI)")

        saved_zones = read_json_safe(ROI_PATH, default={}).get("zones", [])
        if saved_zones:
            st.markdown(
                f'<div class="status-pill status-pill-success" '
                f'style="background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;font-size:0.9rem;">'
                f'✓ <strong>{len(saved_zones)} zone(s)</strong> active on the live feed above.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-pill status-pill-warning" '
                'style="background:#fffbeb;border:1px solid #fde68a;color:#92400e;font-size:0.9rem;">'
                '⚠️ No zones defined. Currently counting EVERYONE the camera sees.'
                '</div>',
                unsafe_allow_html=True,
            )

        st.caption("Open the full-screen editor to draw rectangles with click-and-drag.")

        if st.button("Edit Activity Zones", type="primary", width='stretch', key="open_roi_editor_btn"):
            st.session_state.show_roi_editor = True
            st.rerun()

    with tab_system:
        total_cap = sum(z["capacity"] for z in ZONES.values())
        total_occupied = sum(counts.values())
        st.metric("Total Capacity", f"{total_cap} seats")
        st.metric(
            "Currently Occupied",
            f"{total_occupied} people",
            f"{round(total_occupied / total_cap * 100)}% full",
        )
        st.caption(f"Monitored zones: {len(ZONES)} · Live: Book Common Area 2")

    with tab_password:
        st.markdown("**Change staff password**")
        with st.form("change_pw", clear_on_submit=True):
            old_pw = st.text_input("Current password", type="password", key="pw_current")
            new_pw1 = st.text_input("New password", type="password", key="pw_new1")
            new_pw2 = st.text_input("Confirm new password", type="password", key="pw_new2")
            if st.form_submit_button("Update password", width='stretch'):
                if not check_admin_password(old_pw):
                    st.error("Current password is incorrect.")
                elif not new_pw1:
                    st.error("New password cannot be empty.")
                elif new_pw1 != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    st.session_state.admin_password = new_pw1
                    st.success("Password updated successfully.")

# A hidden Streamlit popover trigger receives clicks from the JS proxy
# in the header. The popover (not a dialog) renders a small dropdown
# panel anchored to its trigger button. The trigger is positioned
# under the header proxy via CSS so Streamlit anchors the panel there.
with st.container(key="hidden_staff_trigger"):
    with st.popover("⋮", width="content"):
        st.markdown(
            '<div style="font-family:Atkinson Hyperlegible Next,sans-serif;font-weight:800;font-size:1rem;'
            'color:#0f172a;margin:0 0 8px 0;letter-spacing:0.3px;">Library Personnel Access</div>',
            unsafe_allow_html=True,
        )
        pw = st.text_input("Password", type="password", key="apw_pop")

        _lock_left = st.session_state.auth_locked_until - time.time()
        if _lock_left > 0:
            st.error(f"Too many attempts. Try again in {int(_lock_left) + 1}s.")
        elif pw and check_admin_password(pw):
            st.session_state.auth_failures = 0
            staff_panel_content()
        elif pw:
            st.session_state.auth_failures += 1
            if st.session_state.auth_failures >= AUTH_MAX_ATTEMPTS:
                st.session_state.auth_locked_until = time.time() + AUTH_LOCKOUT_SECONDS
                st.session_state.auth_failures = 0
                st.error(f"Too many attempts. Locked for {AUTH_LOCKOUT_SECONDS}s.")
            else:
                st.error("✗ Wrong password")


# --- Live dashboard fragment ---
# This block reruns on its own timer (every 5 seconds) without
# touching the rest of the page. Static sections - header, admin
# popover, trend chart, footer - render once per real interaction
# and stay put, which is what kills the page-wide flicker.
@st.fragment(run_every=2.0)
def render_live_dashboard():
    global counts, now, today_str, tomorrow_str
    counts = get_current_counts()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # --- Search + suggestion ---
    col_search, col_result = st.columns([1, 2], vertical_alignment="top")

    with col_search:
        st.markdown(
            "<div class='find-spot-title'>Find a spot</div>"
            "<div class='find-spot-sub'>How many people?</div>",
            unsafe_allow_html=True,
        )
        group_size = st.number_input(
            "Hidden", label_visibility="collapsed",
            min_value=1, max_value=10, value=1, step=1, key="gs",
        )

        advisory = get_peak_advisory()
        for prefix in ("✅ ", "⚠️ ", "📈 ", "📉 "):
            advisory = advisory.replace(prefix, "")
        st.markdown(f'<div class="pk">{advisory}</div>', unsafe_allow_html=True)

    with col_result:
        sug = get_smart_suggestion(counts, group_size)
        if sug:
            p = sug["percentage"]
            sc = "sg" if p < 50 else ("so" if p < 90 else "sr")
            title = f"Best spot for {group_size} {'people' if group_size > 1 else 'person'}"
            body = (
                f"<div style='font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;font-weight:700;opacity:0.8;color:#0f172a;'>{title}</div>"
                f"<div style='font-size:1.75rem;font-weight:800;letter-spacing:-0.5px;line-height:1.2;margin:4px 0;color:#0f172a;'>{sug['zone_name']}</div>"
                f"<div style='font-size:1rem;font-weight:600;background:rgba(0,0,0,0.05);padding:4px 12px;border-radius:12px;display:inline-block;color:#0f172a;'>{sug['available_seats']} seats available</div>"
            )
            st.markdown(f'<div class="sug {sc}">{body}</div>', unsafe_allow_html=True)


    # --- Zone cards ---
    st.markdown(
        '<div class="stl" style="margin-top: 48px;">Quick Status Overview</div><div class="sln"></div>',
        unsafe_allow_html=True,
    )


    # The toggle lives inside an auto-refreshing fragment (run_every=2s).
    # Passing value= on every refresh made Streamlit re-seed the widget each
    # tick, which fought the user's choice and made the toggle flicker/snap
    # back. Seed session state once, bind by key only, and let the key be the
    # single source of truth - no value= and no on_change needed.
    st.session_state.setdefault("show_details_widget", st.session_state.show_details_toggle)
    show_details = st.toggle(
        "Show Detailed Info",
        key="show_details_widget",
    )
    st.session_state.show_details_toggle = show_details


    def _format_upcoming_rows(blist):
        """Compact HTML list of upcoming bookings used inside the detailed card pill."""
        return "".join(
            f'<div class="bk-row">'
            f'<span>{b["start_time"]} – {b["end_time"]}</span>'
            f'<span class="bk-row-name">{b["booked_by"]}</span>'
            f'</div>'
            for b in blist
        )


    def build_detailed_booking_pill():
        """Booking pill for the detailed Discussion Room card. Four states:
        occupied, free-but-reserved-later-today, reserved-future-day, fully-free."""
        current, today_upcoming, future_upcoming = get_booking_state(today_str)

        if current:
            later_label = ""
            later_rows = ""
            if today_upcoming:
                later_label = (
                    f'<span style="font-size:0.7rem;opacity:0.8;margin-top:6px;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.5px;">Later today ({len(today_upcoming)})</span>'
                )
                later_rows = _format_upcoming_rows(today_upcoming)
            return (
                f'<div class="bk br2">🔒 <div class="bk-wrap">'
                f'<span class="bk-label">Currently booked</span>'
                f'<span class="bk-val">{current["booked_by"]} · until {current["end_time"]}</span>'
                f'<span style="font-size:0.72rem;opacity:0.8;margin-top:2px;">{current.get("purpose", "")}</span>'
                f'{later_label}{later_rows}'
                f'</div></div>'
            )

        if today_upcoming:
            count_text = f"{len(today_upcoming)} booking{'s' if len(today_upcoming) != 1 else ''} today"
            return (
                f'<div class="bk bo2">📅 <div class="bk-wrap">'
                f'<span class="bk-label">Available - Scheduled Later</span>'
                f'<span class="bk-val">{count_text}</span>'
                f'{_format_upcoming_rows(today_upcoming)}'
                f'</div></div>'
            )

        if future_upcoming:
            nb = future_upcoming[0]
            if nb["date"] == tomorrow_str:
                date_label = "Tomorrow"
            else:
                try:
                    date_label = datetime.strptime(nb["date"], "%Y-%m-%d").strftime("%b %d")
                except ValueError:
                    date_label = nb.get("date", "")
            return (
                f'<div class="bk bg2">🔓 <div class="bk-wrap">'
                f'<span class="bk-label">Available all day</span>'
                f'<span class="bk-val">Next: {date_label}, {nb["start_time"]} – {nb["end_time"]}</span>'
                f'<span style="font-size:0.72rem;opacity:0.8;margin-top:2px;">{nb["booked_by"]}</span>'
                f'</div></div>'
            )

        return (
            '<div class="bk bg2">🔓 <div class="bk-wrap">'
            '<span class="bk-label">Fully available</span>'
            '<span class="bk-val">No bookings scheduled</span>'
            '</div></div>'
        )


    def build_mini_booking_badge():
        """Single-line booking badge for the compact card variant."""
        current, today_upcoming, _ = get_booking_state(today_str)

        if current:
            more = len(today_upcoming)
            suffix = f" · +{more} more" if more > 0 else ""
            return (
                f'<div class="mini-badge mini-booked">'
                f'🔒 Until {current["end_time"]}{suffix}</div>'
            )

        if today_upcoming:
            nb = today_upcoming[0]
            if len(today_upcoming) == 1:
                text = f'{nb["start_time"]} – {nb["end_time"]}'
            else:
                text = f'{nb["start_time"]} – {nb["end_time"]} · +{len(today_upcoming) - 1} more'
            return (
                f'<div class="mini-badge mini-scheduled">📅 {text}</div>'
            )

        return (
            '<div class="mini-badge mini-open">🔓 Open today</div>'
        )


    def card_detailed(zid, zi, cnt):
        cap = zi["capacity"]
        live_badge = '<span class="lt">LIVE</span>' if zi.get("is_live") else ""
        is_bookable = zi.get("bookable")

        if is_bookable:
            # Booking-only room (Discussion Room). Status is derived from
            # the booking schedule rather than seat count, and the seat
            # numbers are omitted because the room is reserved as a unit.
            current = get_current_booking(st.session_state.bookings)
            if current:
                status = "BOOKED"
                theme = STATUS_THEME["FULL"]
            else:
                status = "AVAILABLE"
                theme = STATUS_THEME["AVAILABLE"]
            emoji = "🔒" if current else "🔓"
            booking_pill = build_detailed_booking_pill()
            return (
                f'<div class="zc" data-zone="{zid}">'
                f'<div class="zr"><span class="zn">{emoji} {zi["name"]}{live_badge}</span>'
                f'<span class="zp {theme["pill"]}">{status}</span></div>'
                f'<div class="zm" style="margin-top:8px;">Reserved as a whole room · booking required</div>'
                f'{booking_pill}'
                f'</div>'
            )

        status, _ = get_status(cnt, cap)
        theme = status_theme(status)
        available = cap - cnt
        pct = round((cnt / cap) * 100) if cap > 0 else 0

        return (
            f'<div class="zc" data-zone="{zid}">'
            f'<div class="zr"><span class="zn">{zi["name"]}{live_badge}</span>'
            f'<span class="zp {theme["pill"]}">{status}</span></div>'
            f'<div class="zb" style="color:{theme["main"]} !important;">{cnt}'
            f'<span style="font-size:1rem;color:#94a3b8;font-weight:600;">/{cap}</span></div>'
            f'<div class="zm">{available} free seats · {pct}% full</div>'
            f'<div class="bar"><div class="fil" style="width:{pct}%;background:{theme["main"]}"></div></div>'
            f'</div>'
        )


    def card_mini(zid, zi, cnt):
        cap = zi["capacity"]
        is_bookable = zi.get("bookable")

        if is_bookable:
            current = get_current_booking(st.session_state.bookings)
            if current:
                theme = STATUS_THEME["FULL"]
                strip_text = f'Booked · until {current["end_time"]}'
            else:
                theme = STATUS_THEME["AVAILABLE"]
                strip_text = "Open now"
            booking_badge = build_mini_booking_badge()
            return (
                f'<div class="zc-mini" data-zone="{zid}">'
                f'<div class="mn-row">'
                f'<span class="mn-name" title="{zi["name"]}">{zi["name"]}</span>'
                f'</div>'
                f'<div class="mn-strip" style="background:{theme["main"]};">{strip_text}</div>'
                f'{booking_badge}'
                f'</div>'
            )

        status, _ = get_status(cnt, cap)
        theme = status_theme(status)
        pct = round((cnt / cap) * 100) if cap > 0 else 0

        return (
            f'<div class="zc-mini" data-zone="{zid}">'
            f'<div class="mn-row">'
            f'<span class="mn-name" title="{zi["name"]}">{zi["name"]}</span>'
            f'<span class="mn-stat">'
            f'<span class="mn-count" style="color:{theme["main"]} !important;">{cnt}'
            f'<span class="mn-cap" style="color:#94a3b8 !important;">/{cap}</span></span>'
            f'<span class="mn-unit">seats in use</span>'
            f'</span>'
            f'</div>'
            f'<div class="mn-bar" style="background:{theme["bg"]};">'
            f'<div class="mn-fill" style="width:{pct}%;background:{theme["main"]};"></div>'
            f'</div>'
            f'<div class="mn-foot"><span class="mn-cta">tap to locate &rsaquo;</span></div>'
            f'</div>'
        )


    if show_details:
        cards_html = '<div class="detailed-grid">' + "".join(
            card_detailed(z, ZONES[z], counts.get(z, 0)) for z in ZONES
        ) + '</div>'
    else:
        cards_html = '<div class="mini-grid">' + "".join(
            card_mini(z, ZONES[z], counts.get(z, 0)) for z in ZONES
        ) + '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


    # --- Floor map ---
    st.markdown(
        '<div id="map-anchor" class="stl" style="margin-top: 48px;">Detailed Floor Map</div><div class="sln"></div>',
        unsafe_allow_html=True,
    )


    def _zone_main_color(zid):
        # Bookable rooms (Discussion Room) take their color from the
        # booking state, not from a seat count.
        if ZONES[zid].get("bookable"):
            current = get_current_booking(st.session_state.bookings)
            return STATUS_THEME["FULL" if current else "AVAILABLE"]["main"]
        status, _ = get_status(counts.get(zid, 0), ZONES[zid]["capacity"])
        return status_theme(status)["main"]


    def _zone_count_label(zid):
        # Bookable rooms show booking state instead of a seat count.
        if ZONES[zid].get("bookable"):
            current = get_current_booking(st.session_state.bookings)
            return "BOOKED" if current else "UNOCCUPIED"
        return f"{counts.get(zid, 0)}/{ZONES[zid]['capacity']}"


    def build_map():
        # Color/label maps. Tuple shape kept for backward-compat
        # with the SVG template's [0] indexing.
        c = {z: (_zone_main_color(z),) for z in ZONES}
        l = {z: _zone_count_label(z) for z in ZONES}

        # --- Layout key (matches the corrected hand-drawn reference) ---
        # Three vertical zones + a right-edge column:
        #   Left strip (x=14-148):   Lounge Ext / Library Lounge / Entrance / Info Desk / Meeting Nook
        #   Mid-left (x=160-380):    Book Common Area 2 (TOP), Book Common Area 1 (BOTTOM)
        #   Open space (x=380-430):  vertical pathway gap with "OPEN SPACE" label
        #   Mid-right (x=430-630):   Book Collections strips, Bleachers, staircases
        #   Right column (x=700-906): LVSG, Discussion, Multimedia, Toilets, Fire Exit, Legend
        # Top + Bottom rows (y=14-52 and y=540-578) hold the Study Carrels.

        # BCA2 tables - table width 190 centered horizontally in rect (x=160-380),
        # so table x=175. Six seats per side, evenly spaced: stride=28, first
        # seat at x=197. Tables start at y=170 (below title at 118, count at
        # 134, and LIVE pill ending at y=155).
        t2 = ""
        for j in range(5):
            y = 170 + j * 26
            t2 += f'<rect x="175" y="{y}" width="190" height="10" fill="#b07c3e" stroke="#7a5526" stroke-width="0.8" rx="2"/>'
            for i in range(6):
                sx = 197 + 28 * i
                t2 += f'<rect x="{sx-1}" y="{y-7}" width="8" height="6" rx="1.5" fill="{c["book_common_2"][0]}"/>'
                t2 += f'<rect x="{sx-1}" y="{y-8}" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'
                t2 += f'<rect x="{sx-1}" y="{y+11}" width="8" height="6" rx="1.5" fill="{c["book_common_2"][0]}"/>'
                t2 += f'<rect x="{sx-1}" y="{y+16}" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'

        # Study Carrels Top + Bottom: 18 cells each. Tables are now thicker
        # (h=5, matches the legend swatch) and stuck to the OUTER WALL of
        # the carrel row - top wall for top carrels, bottom wall for
        # bottom carrels. Seats sit immediately next to the table (1px
        # gap) on the inner side. Title + count text occupy the OPPOSITE
        # side of the rect from the table.
        cell_w = 30
        td = ""
        for i in range(18):
            x_left = 160 + i * cell_w
            if i > 0:
                td += f'<line x1="{x_left}" y1="14" x2="{x_left}" y2="52" stroke="#d9c489" stroke-width="1"/>'
            cx = x_left + cell_w / 2
            # Table stuck to top wall (y=15-20, thickness 5)
            td += f'<rect x="{cx-8}" y="15" width="16" height="6" fill="#b07c3e" stroke="#7a5526" stroke-width="0.7" rx="1"/>'
            # Seat below the table, drawn as a chair with a backrest
            td += f'<rect x="{cx-4}" y="23" width="8" height="6" rx="1.5" fill="{c["carrels_top"][0]}"/>'
            td += f'<rect x="{cx-4}" y="28" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'
        bd = ""
        for i in range(18):
            x_left = 160 + i * cell_w
            if i > 0:
                bd += f'<line x1="{x_left}" y1="540" x2="{x_left}" y2="578" stroke="#d9c489" stroke-width="1"/>'
            cx = x_left + cell_w / 2
            # Seat sits above the table (y=567-571, thickness 4)
            bd += f'<rect x="{cx-4}" y="563" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'
            bd += f'<rect x="{cx-4}" y="565" width="8" height="6" rx="1.5" fill="{c["carrels_bottom"][0]}"/>'
            # Table stuck to bottom wall
            bd += f'<rect x="{cx-8}" y="571" width="16" height="6" fill="#b07c3e" stroke="#7a5526" stroke-width="0.7" rx="1"/>'

        # BCA1 tables - 5 rows centered horizontally in rect (x=160-360).
        # Table width=162, x=179, so margins 19/19 from rect edges. Five
        # seats per side at stride=28, first seat at x=201, so each seat
        # block is symmetrically placed inside the table (22px margins).
        t1l = ""
        for j in range(5):
            y = 380 + j * 26
            t1l += f'<rect x="179" y="{y}" width="162" height="10" fill="#b07c3e" stroke="#7a5526" stroke-width="0.8" rx="2"/>'
            for i in range(5):
                sx = 201 + 28 * i
                t1l += f'<rect x="{sx-1}" y="{y-7}" width="8" height="6" rx="1.5" fill="{c["book_common_1"][0]}"/>'
                t1l += f'<rect x="{sx-1}" y="{y-8}" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'
                t1l += f'<rect x="{sx-1}" y="{y+11}" width="8" height="6" rx="1.5" fill="{c["book_common_1"][0]}"/>'
                t1l += f'<rect x="{sx-1}" y="{y+16}" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'

        # Bottom-right Book Collections - now stretches further right (to x=688)
        # toward the bathroom column, with an extra wavy column on the right
        # edge that gives the corrected map's "books continue around the
        # corner" feel.
        # Bottom-right Book Collections - 11 identical book columns evenly
        # spaced and centered inside the trimmed rectangle (x=372-640).
        # Each book is 12 wide with a 12 gap; 11 books span 252px, leaving
        # an 8px margin on each side. Three rows at y=348/410/472 (62px
        # stride). Trimming the rect width creates a clear pathway between
        # this section and the right column (Toilets / Fire Exit / Legend).
        t1r = ""
        _spines = ("#9c4f42", "#46618a", "#5e7d59", "#a8853f")
        for j in range(3):
            y = 348 + j * 62
            for i in range(11):
                x = 380 + i * 24
                t1r += f'<rect x="{x}" y="{y}" width="12" height="50" fill="#54422a" rx="1.5"/>'
                for k in range(3):
                    t1r += f'<rect x="{x+2}" y="{y+4+k*15}" width="8" height="12" fill="{_spines[(i+j+k) % 4]}" rx="0.8"/>'

        # Lounge Extension furniture (top-left)
        _lec = c["lounge_extension"][0]
        le = ""
        for _sx, _sw in ((24, 38), (68, 28)):
            le += f'<rect x="{_sx}" y="24" width="{_sw}" height="17" rx="4" fill="{_lec}"/>'
            le += f'<rect x="{_sx}" y="24" width="{_sw}" height="5" rx="2.5" fill="#1c1917" opacity="0.22"/>'
            le += f'<rect x="{_sx}" y="29" width="4" height="12" rx="2" fill="#1c1917" opacity="0.16"/>'
            le += f'<rect x="{_sx+_sw-4}" y="29" width="4" height="12" rx="2" fill="#1c1917" opacity="0.16"/>'
        le += '<rect x="24" y="48" width="18" height="8" fill="#b07c3e" stroke="#7a5526" stroke-width="0.7" rx="2"/>'
        le += '<rect x="48" y="48" width="18" height="8" fill="#b07c3e" stroke="#7a5526" stroke-width="0.7" rx="2"/>'
        le += f'<circle cx="83" cy="53" r="4" fill="{_lec}"/>'
        le += f'<circle cx="97" cy="53" r="4" fill="{_lec}"/>'

        # Library Lounge furniture - three lounge clusters + an L-shape at bottom
        _llc = c["library_lounge"][0]
        ll = ""
        for j in range(3):
            y = 140 + j * 42
            ll += f'<rect x="36" y="{y}" width="28" height="13" rx="3" fill="{_llc}"/>'
            ll += f'<rect x="36" y="{y}" width="28" height="4" rx="2" fill="#1c1917" opacity="0.22"/>'
            ll += f'<rect x="28" y="{y+3}" width="6" height="8" rx="2" fill="{_llc}"/>'
            ll += f'<rect x="66" y="{y+3}" width="6" height="8" rx="2" fill="{_llc}"/>'
            ll += f'<circle cx="50" cy="{y-5}" r="3" fill="#b07c3e" stroke="#7a5526" stroke-width="0.6"/>'
            ll += f'<circle cx="50" cy="{y+18}" r="3" fill="#b07c3e" stroke="#7a5526" stroke-width="0.6"/>'
        # L-shaped sofa at the bottom of the Library Lounge (with back edge)
        ll += f'<path d="M26 278 h48 v9 h-48 z M26 266 h9 v21 h-9 z" fill="{_llc}"/>'
        ll += '<path d="M26 278 h48 v2.5 h-48 z M26 266 h2.5 v21 h-2.5 z" fill="#1c1917" opacity="0.2"/>'

        # Bleachers / Reading area camera-side legs along the right edge
        ps = ""
        for j in range(5):
            y = 105 + j * 38
            ps += f'<rect x="660" y="{y}" width="13" height="9" fill="#b07c3e" stroke="#7a5526" stroke-width="0.7" rx="1.5"/>'
            ps += f'<rect x="662" y="{y-8}" width="8" height="6" rx="1.5" fill="{c["bleachers"][0]}"/>'
            ps += f'<rect x="662" y="{y-9}" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'

        # Meeting Nook furniture (bottom-left). Rect spans x=14-150 with
        # center at x=82. Table is 56 wide so x=54 puts its center at 82.
        # Four chairs evenly distributed inside the table: 7px margins
        # on left/right and 6px gaps between chair start positions.
        mn = '<rect x="54" y="494" width="56" height="13" fill="#b07c3e" stroke="#7a5526" stroke-width="0.8" rx="3"/>'
        for cx in (61, 73, 85, 97):
            mn += f'<rect x="{cx-1}" y="486" width="8" height="6" rx="1.5" fill="{c["meeting_nook"][0]}"/>'
            mn += f'<rect x="{cx-1}" y="485" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'
            mn += f'<rect x="{cx-1}" y="509" width="8" height="6" rx="1.5" fill="{c["meeting_nook"][0]}"/>'
            mn += f'<rect x="{cx-1}" y="514" width="8" height="2" rx="1" fill="#1c1917" opacity="0.25"/>'

        svg_code = f"""<style>svg.map-interactive {{ width:100%; height:auto; display:block; background:#fff3c4 !important; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid rgba(148, 163, 184, 0.3); }}.w{{fill:#fffdf7;stroke:#8a7549;stroke-width:1.2;rx:5px;}}.n{{font-family:Atkinson Hyperlegible Next,sans-serif;font-weight:700;font-size:10px;fill:#1c1917;}}.v{{font-family:JetBrains Mono,monospace;font-weight:800;font-size:11.5px;}}.t{{font-family:Atkinson Hyperlegible Next,sans-serif;font-size:6.5px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;fill:#8a8273;}}.g{{font-family:Atkinson Hyperlegible Next,sans-serif;font-size:8px;fill:#7d7460;font-weight:700;}}.p{{font-family:Atkinson Hyperlegible Next,sans-serif;font-size:12px;fill:#cdb978;font-weight:700;letter-spacing:2.5px;}}.map-static rect.w{{fill:#f7edc9;stroke:#c4ad74;}}.map-static .n{{fill:#6f6446;}}text[fill="#64748b"]{{fill:#6f6446;}}.map-zone {{ transition: opacity 0.3s ease, filter 0.3s ease; }}.map-static {{ transition: opacity 0.3s ease; }}svg.map-interactive.dimmed .map-zone {{ opacity: 0.15 !important; }}svg.map-interactive.dimmed .map-static {{ opacity: 0.25 !important; }}svg.map-interactive.dimmed .map-zone.active-zone {{ opacity: 1 !important; filter: drop-shadow(0 0 10px rgba(69,10,10,0.8)) drop-shadow(0 0 20px rgba(69,10,10,0.6)) !important; }}svg.map-interactive.dimmed .map-zone.active-zone rect.w {{ stroke: #450a0a !important; stroke-width: 3 !important; }}</style><svg class="map-interactive" viewBox="0 0 920 598" xmlns="http://www.w3.org/2000/svg"><rect width="920" height="598" fill="#fff3c4" stroke="#8a7549" stroke-width="2.5" rx="16"/><g id="zone-carrels_top" class="map-zone"><rect x="160" y="14" width="540" height="38" class="w" fill="#ffffff"/>{td}<text x="430" y="36" text-anchor="middle" class="n" font-size="9">Study Carrels (Top)</text><text x="430" y="48" text-anchor="middle" class="v" font-size="9" fill="{c['carrels_top'][0]}">{l['carrels_top']}</text></g><g id="zone-lounge_extension" class="map-zone"><rect x="14" y="14" width="136" height="66" class="w" fill="#ffffff"/>{le}<text x="82" y="76" text-anchor="middle" font-family="Atkinson Hyperlegible Next,sans-serif" font-weight="800"><tspan class="n" font-size="9">Lounge Ext. · </tspan><tspan font-size="12" fill="{c['lounge_extension'][0]}">{l['lounge_extension']}</tspan></text></g><g class="map-static"><rect x="160" y="70" width="220" height="20" class="w" fill="#ffffff"/><text x="270" y="84" text-anchor="middle" class="n">Book Collections</text></g><g id="zone-library_lounge" class="map-zone"><rect x="14" y="90" width="136" height="210" class="w" fill="#ffffff"/>{ll}<text x="82" y="106" text-anchor="middle" class="n">Library Lounge</text><text x="82" y="120" text-anchor="middle" class="v" fill="{c['library_lounge'][0]}" font-size="11.5">{l['library_lounge']}</text></g><g id="zone-book_common_2" class="map-zone"><rect x="160" y="100" width="220" height="200" class="w" fill="#ffffff"/>{t2}<text x="270" y="118" text-anchor="middle" class="n" font-size="10">Book Common Area 2</text><text x="270" y="134" text-anchor="middle" class="v" font-size="13" fill="{c['book_common_2'][0]}">{l['book_common_2']}</text><rect x="246" y="140" width="48" height="15" rx="4" fill="#ef4444"/><circle cx="256" cy="147.5" r="2.5" fill="#fff" opacity="0.9"><animate attributeName="opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite"/></circle><text x="276" y="151" text-anchor="middle" fill="#fff" font-family="JetBrains Mono,monospace" font-size="7.5" font-weight="800">LIVE</text></g><g class="map-static"><rect x="430" y="80" width="200" height="32" class="w" fill="#ffffff"/><text x="530" y="99" text-anchor="middle" class="n">Book Collections</text><rect x="430" y="118" width="200" height="20" class="w" fill="#ffffff" stroke-dasharray="3 3" stroke="#94a3b8"/><g class="ic" fill="#8a7549"><rect x="440" y="134" width="5" height="3"/><rect x="445" y="131" width="5" height="6"/><rect x="450" y="128" width="5" height="9"/><rect x="455" y="125" width="5" height="12"/></g><text x="530" y="132" text-anchor="middle" fill="#64748b" font-family="Atkinson Hyperlegible Next,sans-serif" font-size="7.5" font-weight="700">&#9660; Staircase Down</text></g><g id="zone-bleachers" class="map-zone"><rect x="430" y="144" width="200" height="72" class="w" fill="#ffffff"/><line x1="438" y1="157" x2="622" y2="157" stroke="#d9c489" stroke-width="1"/><line x1="438" y1="167" x2="622" y2="167" stroke="#d9c489" stroke-width="1"/><line x1="438" y1="177" x2="622" y2="177" stroke="#d9c489" stroke-width="1"/><line x1="438" y1="187" x2="622" y2="187" stroke="#d9c489" stroke-width="1"/><line x1="438" y1="197" x2="622" y2="197" stroke="#d9c489" stroke-width="1"/><rect x="442" y="159" width="26" height="5" fill="{c['bleachers'][0]}" rx="1"/><rect x="488" y="169" width="40" height="5" fill="{c['bleachers'][0]}" rx="1"/><rect x="580" y="179" width="22" height="5" fill="{c['bleachers'][0]}" rx="1"/><text x="530" y="182" text-anchor="middle" class="n">Bleachers / Reading</text><text x="530" y="197" text-anchor="middle" class="v" fill="{c['bleachers'][0]}">{l['bleachers']}</text></g><g class="map-static"><rect x="430" y="222" width="200" height="20" class="w" fill="#ffffff" stroke-dasharray="3 3" stroke="#94a3b8"/><g class="ic" fill="#8a7549"><rect x="440" y="238" width="5" height="3"/><rect x="445" y="235" width="5" height="6"/><rect x="450" y="232" width="5" height="9"/><rect x="455" y="229" width="5" height="12"/></g><text x="530" y="236" text-anchor="middle" fill="#64748b" font-family="Atkinson Hyperlegible Next,sans-serif" font-size="7.5" font-weight="700">&#9650; Staircase Up to 2F</text><rect x="430" y="248" width="200" height="36" class="w" fill="#ffffff"/><text x="530" y="270" text-anchor="middle" class="n">Book Collections</text>{ps}<text x="530" y="70" text-anchor="middle" class="p">PATHWAY</text><text x="270" y="320" text-anchor="middle" class="p" font-size="10">PATHWAY</text><text x="530" y="312" text-anchor="middle" class="p">PATHWAY</text><text x="405" y="200" text-anchor="middle" class="p" font-size="11" transform="rotate(-90,405,200)">OPEN SPACE</text><g><circle cx="404" cy="266" r="5.5" fill="#557a48"/><circle cx="399" cy="261" r="4" fill="#6b9159"/><circle cx="409" cy="261" r="4" fill="#4c7340"/></g><g><circle cx="670" cy="332" r="5.5" fill="#557a48"/><circle cx="665" cy="327" r="4" fill="#6b9159"/><circle cx="675" cy="327" r="4" fill="#4c7340"/></g><text x="670" y="423" text-anchor="middle" class="p" font-size="11" transform="rotate(-90,670,423)">PATHWAY</text></g><g class="map-static"><rect x="700" y="14" width="206" height="42" class="w" fill="#ffffff"/><text x="803" y="35" text-anchor="middle" class="n">LVSG Room</text><text x="803" y="48" text-anchor="middle" class="t">(No seating)</text></g><g id="zone-discussion_room" class="map-zone"><rect x="700" y="64" width="206" height="80" class="w" fill="#ffffff"/><text x="803" y="90" text-anchor="middle" class="n">Discussion Room</text><text x="803" y="108" text-anchor="middle" class="v" font-size="13" fill="{c['discussion_room'][0]}">{l['discussion_room']}</text><text x="803" y="128" text-anchor="middle" class="t">Booking required</text></g><g id="zone-multimedia_room" class="map-zone"><rect x="700" y="152" width="206" height="80" class="w" fill="#ffffff"/><text x="803" y="178" text-anchor="middle" class="n">Multimedia Room</text><text x="803" y="196" text-anchor="middle" class="v" font-size="13" fill="{c['multimedia_room'][0]}">{l['multimedia_room']}</text></g><g class="map-static"><rect x="700" y="240" width="206" height="68" class="w" fill="#ffffff"/><g class="ic" fill="#7d7460"><circle cx="803" cy="252" r="3.4"/><path d="M803 257 l6.5 11 h-13 z"/></g><text x="803" y="278" text-anchor="middle" class="g">Female Toilet</text><rect x="700" y="316" width="206" height="68" class="w" fill="#ffffff"/><g class="ic" fill="#7d7460"><circle cx="803" cy="328" r="3.4"/><rect x="799" y="333" width="8" height="11" rx="2.5"/></g><text x="803" y="354" text-anchor="middle" class="g">Male Toilet</text><rect x="700" y="392" width="206" height="60" class="w" fill="#ffffff"/><text x="803" y="426" text-anchor="middle" class="g">Fire Exit</text><rect x="14" y="318" width="30" height="120" class="w" fill="#ffffff"/><rect x="16" y="320" width="26" height="56" fill="#10b981" opacity="0.10" rx="1.5"/><rect x="16" y="380" width="26" height="56" fill="#ef4444" opacity="0.10" rx="1.5"/><g stroke="#10b981" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none"><polyline points="22,326 32,333 22,340"><animate attributeName="opacity" values="0.2;1;0.2" dur="1.8s" repeatCount="indefinite" begin="0s"/></polyline><polyline points="22,346 32,353 22,360"><animate attributeName="opacity" values="0.2;1;0.2" dur="1.8s" repeatCount="indefinite" begin="0.3s"/></polyline></g><g stroke="#ef4444" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none"><polyline points="32,396 22,403 32,410"><animate attributeName="opacity" values="0.2;1;0.2" dur="1.8s" repeatCount="indefinite" begin="0.6s"/></polyline><polyline points="32,416 22,423 32,430"><animate attributeName="opacity" values="0.2;1;0.2" dur="1.8s" repeatCount="indefinite" begin="0.9s"/></polyline></g><text x="29" y="378" text-anchor="middle" class="n" font-size="7" letter-spacing="1.5" transform="rotate(-90,29,378)">ENTRANCE</text><rect x="52" y="340" width="98" height="76" class="w" fill="#ffffff"/><circle cx="101" cy="368" r="18" fill="#fffdf7" stroke="#8a7549" stroke-width="1.2"/><text x="101" y="372" text-anchor="middle" fill="#ef4444" font-family="Atkinson Hyperlegible Next,sans-serif" font-size="9" font-weight="800">O</text><rect x="78" y="389" width="46" height="5" rx="2.5" fill="#b07c3e" stroke="#7a5526" stroke-width="0.7"/><text x="101" y="407" text-anchor="middle" class="g">Info Desk</text></g><g id="zone-book_common_1" class="map-zone"><rect x="160" y="330" width="200" height="198" class="w" fill="#ffffff"/>{t1l}<text x="260" y="346" text-anchor="middle" class="n">Book Common Area 1</text><text x="260" y="361" text-anchor="middle" class="v" fill="{c['book_common_1'][0]}">{l['book_common_1']}</text></g><g class="map-static"><rect x="372" y="318" width="268" height="210" class="w" fill="#ffffff"/>{t1r}<text x="506" y="338" text-anchor="middle" class="n">Book Collections</text></g><g id="zone-meeting_nook" class="map-zone"><rect x="14" y="448" width="136" height="80" class="w" fill="#ffffff"/>{mn}<text x="82" y="464" text-anchor="middle" class="n">Meeting Nook</text><text x="82" y="478" text-anchor="middle" class="v" fill="{c['meeting_nook'][0]}" font-size="11.5">{l['meeting_nook']}</text></g><g id="zone-carrels_bottom" class="map-zone"><rect x="160" y="540" width="540" height="38" class="w" fill="#ffffff"/>{bd}<text x="430" y="552" text-anchor="middle" class="n" font-size="9">Study Carrels (Bottom)</text><text x="430" y="562" text-anchor="middle" class="v" font-size="9" fill="{c['carrels_bottom'][0]}">{l['carrels_bottom']}</text></g><g class="map-static"><rect x="700" y="460" width="206" height="68" class="w" fill="#ffffff"/><text x="710" y="475" class="n">Legend</text><rect x="710" y="485" width="7" height="2" rx="1" fill="#1c1917" opacity="0.25"/><rect x="710" y="486" width="7" height="5" rx="1.5" fill="#10b981"/><text x="722" y="491" class="t">Seat - available</text><rect x="710" y="497" width="7" height="2" rx="1" fill="#1c1917" opacity="0.25"/><rect x="710" y="498" width="7" height="5" rx="1.5" fill="#f59e0b"/><text x="722" y="503" class="t">Seat - busy</text><rect x="710" y="509" width="7" height="2" rx="1" fill="#1c1917" opacity="0.25"/><rect x="710" y="510" width="7" height="5" rx="1.5" fill="#ef4444"/><text x="722" y="515" class="t">Seat - full</text><rect x="806" y="486" width="12" height="6" fill="#b07c3e" stroke="#7a5526" stroke-width="0.6" rx="1"/><text x="823" y="491" class="t">Table</text><rect x="806" y="498" width="12" height="7" fill="#54422a" rx="1"/><rect x="808" y="499.5" width="3.5" height="4" fill="#9c4f42" rx="0.5"/><rect x="812.5" y="499.5" width="3.5" height="4" fill="#46618a" rx="0.5"/><text x="823" y="504" class="t">Books</text><rect x="806" y="510" width="20" height="7" rx="3" fill="#ef4444"/><text x="816" y="516" text-anchor="middle" fill="#fff" font-family="JetBrains Mono,monospace" font-size="4.5" font-weight="800">LIVE</text><text x="830" y="516" class="t">Live camera</text></g></svg>"""
        return "".join(line.strip() for line in svg_code.splitlines())


    st.markdown(build_map(), unsafe_allow_html=True)

    # Floor 2 toggle - clicking the button flips a session-state flag, so
    # clicking it once shows the coming-soon message and clicking again
    # dismisses it.
    st.session_state.setdefault("show_floor2", False)
    btn_label = "↑  Hide Floor 2 Map" if st.session_state.show_floor2 else "↓  Switch to Floor 2 Map"
    if st.button(btn_label, width='stretch', key="floor2_toggle"):
        st.session_state.show_floor2 = not st.session_state.show_floor2
        # This button lives inside an auto-refreshing fragment (run_every).
        # A bare st.rerun() is disallowed there and crashes; scope="fragment"
        # reruns just this fragment so the toggle updates instantly and safely.
        st.rerun(scope="fragment")
    if st.session_state.show_floor2:
        st.markdown(
            '<div class="cs"><div style="font-size:2.5rem; margin-bottom:12px;">🚧</div>'
            '<div class="cs-title">Floor 2 Map Coming Soon</div>'
            '<div class="cs-sub">Our team is currently mapping the second-floor layout.</div></div>',
            unsafe_allow_html=True,
        )



render_live_dashboard()

# --- Trend chart ---
# Skip this heavy Plotly rebuild while the ROI editor dialog is open. When
# the dialog is up, the user is focused on the canvas; rebuilding the chart
# (and the big SVG map in the fragment) underneath the modal on every
# interaction is what made the editor feel sluggish to open. A lightweight
# placeholder keeps the layout stable without the render cost.
if st.session_state.get("show_roi_editor"):
    st.markdown(
        '<div class="stl" style="margin-top: 48px;">Occupancy Trend - Today</div>'
        '<div class="sln"></div>'
        '<div style="height:280px;display:flex;align-items:center;justify-content:center;'
        'color:#94a3b8;font-family:Atkinson Hyperlegible Next,sans-serif;font-size:0.9rem;">'
        'Chart paused while the zone editor is open…</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="stl" style="margin-top: 48px;">Occupancy Trend - Today</div><div class="sln"></div>',
        unsafe_allow_html=True,
    )
    hours, occ, _ = get_cached_historical()
    total_capacity = sum(z["capacity"] for z in ZONES.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=occ, mode='lines+markers',
        line=dict(color='#7a1420', width=3, shape='spline'),
        marker=dict(size=8, color='#7a1420', line=dict(width=2, color='#fffdf9')),
        fill='tozeroy', fillcolor='rgba(122,20,32,0.07)',
        hovertemplate='<b>%{x}</b><br>%{y} people<extra></extra>',
    ))
    fig.add_hline(
        y=total_capacity * 0.5, line_dash="dot", line_color="#10b981", line_width=1.5,
        annotation_text="Busy Threshold", annotation_font_color="#10b981",
        annotation_font_size=11, annotation_position="top left",
    )
    fig.add_hline(
        y=total_capacity * 0.9, line_dash="dot", line_color="#ef4444", line_width=1.5,
        annotation_text="Full Capacity", annotation_font_color="#ef4444",
        annotation_font_size=11, annotation_position="top left",
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Atkinson Hyperlegible Next,sans-serif", color="#6b6457", size=12),
        xaxis=dict(gridcolor='rgba(120, 113, 108, 0.12)', showline=True, linecolor='rgba(120, 113, 108, 0.3)', fixedrange=True, tickfont=dict(weight=600)),
        yaxis=dict(gridcolor='rgba(120, 113, 108, 0.12)', title="Total Occupants", range=[0, total_capacity + 10], showline=True, linecolor='rgba(120, 113, 108, 0.3)', fixedrange=True, tickfont=dict(weight=600)),
        height=280, margin=dict(l=48, r=24, t=16, b=40), showlegend=False, hovermode='x unified',
    )
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


# --- Footer ---
st.markdown(
    f'<div class="ftr"><b>LOC-LOC</b> v2.2.0 · Real-Time Library Occupancy Monitoring<br>'
    f'LOKASYON - Alis, Carvajal, Dela Cruz, Li, Lumangcas<br>'
    f'Lyceum of the Philippines University Manila · {now.year}</div>',
    unsafe_allow_html=True,
)


# --- JS bridge: card->map highlighting + admin popover repositioning ---
js_bridge = """
<script>
const pDoc = window.parent.document;
const pWin = window.parent;

// --- Staff button proxy in the fixed header ---
// The proxy in the header forwards clicks to a hidden Streamlit button.
// Hiding the trigger button is handled by CSS (.st-key-hidden_staff_trigger)
// so it never flashes on screen. JS only handles the click forwarding.
function setupHeaderButtons() {
    const slot = pDoc.getElementById('pinned-staff-slot');
    if (!slot) return;

    // Theme toggle button - lives in localStorage so it persists across
    // reloads. Toggles data-theme="dark" on <body>, which a dedicated
    // CSS block recolors to a dark palette.
    if (!pDoc.getElementById('theme-toggle-btn')) {
        const themeBtn = pDoc.createElement('button');
        themeBtn.id = 'theme-toggle-btn';
        themeBtn.type = 'button';
        themeBtn.setAttribute('aria-label', 'Toggle dark mode');
        const applyTheme = (mode) => {
            pDoc.documentElement.setAttribute('data-theme', mode);
            pDoc.body.setAttribute('data-theme', mode);
            themeBtn.textContent = mode === 'dark' ? '\u2600' : '\u263D';
            themeBtn.title = mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
        };
        const stored = (() => {
            try { return window.parent.localStorage.getItem('locloc-theme'); }
            catch (e) { return null; }
        })() || 'light';
        applyTheme(stored);
        themeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const next = pDoc.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            applyTheme(next);
            try { window.parent.localStorage.setItem('locloc-theme', next); }
            catch (err) {}
            // Re-darkify canvas iframes so any open dialog updates immediately.
            if (typeof darkifyCanvasIframes === 'function') {
                setTimeout(darkifyCanvasIframes, 50);
            }
        });
        slot.appendChild(themeBtn);
    }

    // Staff proxy button - visible 3-dots in the header, forwards
    // clicks to Streamlit's hidden popover trigger.
    if (!pDoc.getElementById('staff-proxy-btn')) {
        const proxy = pDoc.createElement('button');
        proxy.id = 'staff-proxy-btn';
        proxy.type = 'button';
        proxy.title = 'Library personnel access';
        proxy.setAttribute('aria-label', 'Library personnel access');
        proxy.textContent = '\u22EE';
        // Stop the mousedown event from reaching Streamlit's outside-click
        // listener BEFORE the popover toggle fires. Without this, clicking
        // the proxy when the popover is open: (1) Streamlit closes via
        // outside-click then (2) our trigger.click() reopens it on the
        // same tick -> looks like nothing happened
        proxy.addEventListener('mousedown', (e) => {
            e.stopPropagation();
        });
        proxy.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const trigger = pDoc.querySelector('.st-key-hidden_staff_trigger [data-testid="stPopover"] button');
            if (trigger) trigger.click();
        });
        slot.appendChild(proxy);
    }
}
setTimeout(setupHeaderButtons, 50);
setTimeout(setupHeaderButtons, 200);
setTimeout(setupHeaderButtons, 600);
// (Re-runs after DOM changes via the single shared observer below.)

// --- Map zoom / pan (scoped to the floor-plan SVG only) ---
// Lets you zoom and pan the map without affecting the rest of the
// dashboard - useful on a phone. Buttons and double-tap work everywhere;
// the mouse wheel zooms on desktop; one finger pans and two fingers pinch
// once you're zoomed in. Page scrolling is preserved at 1x so the map
// doesn't trap the scroll. Zoom state lives on the parent window so it
// survives the dashboard's periodic refresh.
function setupMapZoom() {
    const svg = pDoc.querySelector('svg.map-interactive');
    if (!svg) return;
    const view = svg.closest('[data-testid="stMarkdownContainer"]') || svg.parentElement;
    const frame = svg.closest('[data-testid="element-container"]') || view;
    if (!view || !frame) return;

    const Z = pWin._locMapZoom = pWin._locMapZoom || { scale: 1, x: 0, y: 0 };

    view.style.overflow = 'hidden';
    view.style.borderRadius = '16px';
    view.style.position = 'relative';
    svg.style.transformOrigin = '0 0';
    svg.style.willChange = 'transform';

    function clamp() {
        if (Z.scale <= 1) { Z.scale = 1; Z.x = 0; Z.y = 0; return; }
        const w = view.clientWidth, h = svg.clientHeight || view.clientHeight;
        Z.x = Math.min(0, Math.max(w - w * Z.scale, Z.x));
        Z.y = Math.min(0, Math.max(h - h * Z.scale, Z.y));
    }
    function apply() {
        clamp();
        svg.style.transform = 'translate(' + Z.x + 'px,' + Z.y + 'px) scale(' + Z.scale + ')';
        svg.style.cursor = Z.scale > 1 ? 'grab' : 'default';
        // Trap touch scrolling only while zoomed; at 1x let the page scroll.
        view.style.touchAction = Z.scale > 1 ? 'none' : 'pan-y';
        const hint = frame.querySelector('.map-zoom-hint');
        if (hint) hint.style.opacity = Z.scale > 1 ? '0' : '0.85';
    }
    pWin._locApplyMapZoom = apply;
    function zoomAt(cx, cy, factor) {
        const ns = Math.min(5, Math.max(1, Z.scale * factor));
        const k = ns / Z.scale;
        Z.x = cx - (cx - Z.x) * k;
        Z.y = cy - (cy - Z.y) * k;
        Z.scale = ns;
        apply();
    }
    pWin._locZoomStep = function(dir) {
        const w = view.clientWidth, h = svg.clientHeight || view.clientHeight;
        if (dir === 'reset') { Z.scale = 1; Z.x = 0; Z.y = 0; apply(); }
        else zoomAt(w / 2, h / 2, dir === 'in' ? 1.4 : 1 / 1.4);
    };

    if (!svg.dataset.zoomBound) {
        svg.dataset.zoomBound = '1';
        const pts = new Map();
        let startDist = 0, startScale = 1, startMid = null, startXY = null, panStart = null, moved = false;

        svg.addEventListener('wheel', function(e) {
            e.preventDefault();
            const r = view.getBoundingClientRect();
            zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
        }, { passive: false });

        svg.addEventListener('pointerdown', function(e) {
            try { svg.setPointerCapture(e.pointerId); } catch (err) {}
            pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
            moved = false;
            if (pts.size === 1) {
                panStart = { x: e.clientX, y: e.clientY, ox: Z.x, oy: Z.y };
                if (Z.scale > 1) svg.style.cursor = 'grabbing';
            } else if (pts.size === 2) {
                const a = Array.from(pts.values());
                startDist = Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
                startScale = Z.scale;
                const r = view.getBoundingClientRect();
                startMid = { x: (a[0].x + a[1].x) / 2 - r.left, y: (a[0].y + a[1].y) / 2 - r.top };
                startXY = { x: Z.x, y: Z.y };
            }
        });
        svg.addEventListener('pointermove', function(e) {
            if (!pts.has(e.pointerId)) return;
            pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (pts.size >= 2 && startDist > 0) {
                const a = Array.from(pts.values());
                const dist = Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
                const ns = Math.min(5, Math.max(1, startScale * (dist / startDist)));
                const k = ns / startScale;
                Z.x = startMid.x - (startMid.x - startXY.x) * k;
                Z.y = startMid.y - (startMid.y - startXY.y) * k;
                Z.scale = ns;
                moved = true;
                apply();
            } else if (panStart && Z.scale > 1) {
                Z.x = panStart.ox + (e.clientX - panStart.x);
                Z.y = panStart.oy + (e.clientY - panStart.y);
                if (Math.abs(e.clientX - panStart.x) + Math.abs(e.clientY - panStart.y) > 4) moved = true;
                apply();
            }
        });
        function endPtr(e) {
            pts.delete(e.pointerId);
            if (pts.size < 2) startDist = 0;
            if (pts.size === 0) { panStart = null; svg.style.cursor = Z.scale > 1 ? 'grab' : 'default'; }
        }
        svg.addEventListener('pointerup', endPtr);
        svg.addEventListener('pointercancel', endPtr);
        // Swallow the click that ends a pan so it doesn't clear the highlight.
        svg.addEventListener('click', function(e) {
            if (moved) { e.stopPropagation(); e.preventDefault(); moved = false; }
        }, true);
        svg.addEventListener('dblclick', function(e) {
            e.preventDefault(); e.stopPropagation();
            const r = view.getBoundingClientRect();
            if (Z.scale >= 2.5) pWin._locZoomStep('reset');
            else zoomAt(e.clientX - r.left, e.clientY - r.top, 1.8);
        }, true);
    }

    // Controls + hint live on the persistent element-container so they
    // survive the markdown rebuild; re-injected only if missing.
    if (!frame.querySelector('.map-zoom-ctrl')) {
        frame.style.position = 'relative';
        const ctrl = pDoc.createElement('div');
        ctrl.className = 'map-zoom-ctrl';
        ctrl.innerHTML =
            '<button type="button" data-z="in" aria-label="Zoom in">+</button>' +
            '<button type="button" data-z="reset" aria-label="Reset zoom">\u2922</button>' +
            '<button type="button" data-z="out" aria-label="Zoom out">\u2212</button>';
        ctrl.addEventListener('pointerdown', function(e) { e.stopPropagation(); });
        ctrl.addEventListener('click', function(e) {
            const b = e.target.closest('button'); if (!b) return;
            e.stopPropagation();
            if (pWin._locZoomStep) pWin._locZoomStep(b.getAttribute('data-z'));
        });
        frame.appendChild(ctrl);
        const hint = pDoc.createElement('div');
        hint.className = 'map-zoom-hint';
        hint.textContent = 'Pinch / scroll to zoom';
        frame.appendChild(hint);
    }
    apply();
}
setTimeout(setupMapZoom, 120);
setTimeout(setupMapZoom, 400);
setTimeout(setupMapZoom, 900);
// (Re-runs after DOM changes via the single shared observer below.)

// Reach into the canvas component iframe and set its body background
// to match the dark theme. The iframe is same-origin (Streamlit
// component) so contentDocument access is allowed.
function darkifyCanvasIframes() {
    const isDark = pDoc.documentElement.getAttribute('data-theme') === 'dark';
    pDoc.querySelectorAll('[role="dialog"] iframe').forEach(iframe => {
        try {
            const idoc = iframe.contentDocument;
            if (!idoc || !idoc.body) return;
            idoc.body.style.backgroundColor = isDark ? '#1e293b' : '';
            // Also set on the html and any direct wrapper divs
            if (idoc.documentElement) {
                idoc.documentElement.style.backgroundColor = isDark ? '#1e293b' : '';
            }
        } catch (e) { /* cross-origin iframe - can't help */ }
    });
}
setTimeout(darkifyCanvasIframes, 200);
setTimeout(darkifyCanvasIframes, 600);
setTimeout(darkifyCanvasIframes, 1500);

// --- one shared observer ---
// before there were 3 separate observers (header buttons, map zoom,
// canvas dark mode) all firing on every DOM change, and the live
// refresh changes the DOM every 2s. one observer with a 150ms debounce
// does the same job way cheaper
if (pDoc._locSharedObs) pDoc._locSharedObs.disconnect();
let _locSharedPending = false;
pDoc._locSharedObs = new MutationObserver(() => {
    if (_locSharedPending) return;
    _locSharedPending = true;
    setTimeout(() => {
        _locSharedPending = false;
        setupHeaderButtons();
        setupMapZoom();
        darkifyCanvasIframes();
        applyZoneSelection();
    }, 150);
});
pDoc._locSharedObs.observe(pDoc.body, { childList: true, subtree: true });

if (pDoc._locClickHandler) {
    pDoc.removeEventListener('click', pDoc._locClickHandler);
}

// --- Smooth live camera feed ---
// Preload the next frame into a detached Image() and only swap the visible
// <img> once it has fully decoded. That removes the blank flash you get from
// reassigning src directly. An in-flight guard prevents request pile-up, and
// polling at ~5fps (vs the old 1fps) makes the feed look like video.
if (pDoc._locFeedTimer) {
    clearInterval(pDoc._locFeedTimer);
}
pDoc._locFeedLoading = false;
pDoc._locFeedTimer = setInterval(() => {
    const img = pDoc.getElementById('loc-live-feed');
    if (!img || pDoc._locFeedLoading) return;
    pDoc._locFeedLoading = true;
    const next = new Image();
    const done = () => { pDoc._locFeedLoading = false; };
    next.src = './app/static/latest_frame.jpg?t=' + Date.now();
    if (next.decode) {
        // decode() prepares the jpg in the background so swapping the
        // image never blocks clicks or scrolling
        next.decode().then(() => { img.src = next.src; done(); }).catch(done);
    } else {
        next.onload = () => { img.src = next.src; done(); };
        next.onerror = done;
    }
}, 200);

// put a bouncing arrow above the zone. zones near the top edge get the
// arrow flipped (points up from below) so it does not get cut off
function placeZoneArrow(zoneId) {
    const svg = pDoc.querySelector('svg.map-interactive');
    if (!svg) return;
    const old = svg.querySelector('.loc-arrow');
    if (old) old.remove();
    if (!zoneId) return;
    const zone = pDoc.getElementById('zone-' + zoneId);
    if (!zone) return;
    let bb;
    try { bb = zone.getBBox(); } catch (err) { return; }
    const cx = bb.x + bb.width / 2;
    const flip = bb.y < 42;
    const ay = flip ? (bb.y + bb.height) : bb.y;
    const g = pDoc.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'loc-arrow');
    g.setAttribute('transform', 'translate(' + cx + ',' + ay + ')' + (flip ? ' rotate(180)' : ''));
    g.innerHTML = '<ellipse cx="0" cy="-4" rx="9" ry="2.6"></ellipse>' +
        '<g class="loc-arrow-inner"><polygon points="-12,-33 12,-33 0,-9"></polygon></g>';
    svg.appendChild(g);
}

// re-apply the saved selection (dim, highlight, arrow, card ring).
// the live refresh rebuilds the map every 2s which used to erase the
// highlight, so the observer calls this to put it back. exits early if
// everything is already applied so it cant loop with the observer
function applyZoneSelection() {
    const svg = pDoc.querySelector('svg.map-interactive');
    if (!svg) return;
    const zoneId = pDoc._locSelectedZone;
    const arrow = svg.querySelector('.loc-arrow');
    if (!zoneId) {
        if (!svg.classList.contains('dimmed') && !arrow) return;
        svg.classList.remove('dimmed');
        if (arrow) arrow.remove();
        pDoc.querySelectorAll('.map-zone').forEach(z => z.classList.remove('active-zone'));
        return;
    }
    const target = pDoc.getElementById('zone-' + zoneId);
    if (svg.classList.contains('dimmed') && target &&
        target.classList.contains('active-zone') && arrow) {
        return;
    }
    svg.classList.add('dimmed');
    pDoc.querySelectorAll('.map-zone').forEach(z => z.classList.remove('active-zone'));
    if (target) target.classList.add('active-zone');
    placeZoneArrow(zoneId);
    pDoc.querySelectorAll('.zc, .zc-mini').forEach(c => {
        c.classList.toggle('selected', c.getAttribute('data-zone') === zoneId);
    });
}

pDoc._locClickHandler = function(e) {
    const card = e.target.closest('.zc, .zc-mini');
    const svg = pDoc.querySelector('svg.map-interactive');
    if (!svg) return;

    if (card) {
        const zoneId = card.getAttribute('data-zone');
        if (!zoneId) return;

        pDoc.querySelectorAll('.zc, .zc-mini').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        pDoc._locSelectedZone = zoneId;
        applyZoneSelection();

        // Reset any map zoom so the highlighted zone is actually in view.
        if (pWin._locZoomStep) pWin._locZoomStep('reset');

        const mapAnchor = pDoc.getElementById('map-anchor');
        if (mapAnchor) mapAnchor.scrollIntoView({ behavior: 'smooth', block: 'start' });

        return;
    }

    // clicked outside any card - only dismiss if it's not a real interactive control or the map itself
    const isTrueInteractive = e.target.closest(
        'button, input, select, textarea, ' +
        '[data-testid="stPopover"], [data-testid="stCheckbox"], ' +
        '[role="button"], a, label'
    );
    const isOnMap = e.target.closest('svg.map-interactive');

    if (!isTrueInteractive && !isOnMap) {
        pDoc._locSelectedZone = null;
        applyZoneSelection();
        pDoc.querySelectorAll('.zc, .zc-mini').forEach(c => c.classList.remove('selected'));
    }
};
pDoc.addEventListener('click', pDoc._locClickHandler);

// --- card glow effect ---
// one pointermove listener for the whole page. card children have
// pointer-events none so e.target is the card itself when hovering.
// throttled with requestAnimationFrame so its max one update per frame
if (pDoc._locGlowHandler) {
    pDoc.removeEventListener('pointermove', pDoc._locGlowHandler);
}
let _locGlowRaf = null, _locGlowEv = null;
pDoc._locGlowHandler = function(e) {
    _locGlowEv = e;
    if (_locGlowRaf) return;
    _locGlowRaf = pWin.requestAnimationFrame(() => {
        _locGlowRaf = null;
        const t = _locGlowEv.target;
        const card = t && t.closest ? t.closest('.zc, .zc-mini') : null;
        if (!card) return;
        const r = card.getBoundingClientRect();
        card.style.setProperty('--mx', (_locGlowEv.clientX - r.left) + 'px');
        card.style.setProperty('--my', (_locGlowEv.clientY - r.top) + 'px');
    });
};
pDoc.addEventListener('pointermove', pDoc._locGlowHandler, { passive: true });
</script>
"""
components.html(js_bridge, height=0, width=0)