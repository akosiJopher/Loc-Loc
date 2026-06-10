"""
LOC-LOC Data Source Module

All the zone info and the simulated numbers are here. The live camera
count from counts.json is also read here. The dashboard (app.py) just
calls the functions in this file, so connecting more real cameras later
only needs changes here.
"""

import random
import time
from datetime import datetime, timedelta

# ============================================================
# ZONE CONFIGURATION
# ============================================================

ZONES = {
    "book_common_1": {
        "name": "Book Common Area 1",
        "capacity": 42,
        "type": "study",
        "description": "Large group tables near entrance",
        "group_friendly": True,
        "max_group_size": 8,
    },
    "book_common_2": {
        "name": "Book Common Area 2",
        "capacity": 50,
        "type": "study",
        "description": "Large group tables - center area",
        "group_friendly": True,
        "max_group_size": 8,
        "is_live": True,
    },
    "library_lounge": {
        "name": "Library Lounge",
        "capacity": 16,
        "type": "lounge",
        "description": "Lounge-style seating",
        "group_friendly": True,
        "max_group_size": 4,
    },
    "lounge_extension": {
        "name": "Lounge Extension",
        "capacity": 12,
        "type": "lounge",
        "description": "Additional lounge seating - top left",
        "group_friendly": True,
        "max_group_size": 4,
    },
    "bleachers": {
        "name": "Bleachers / Reading Area",
        "capacity": 20,
        "type": "study",
        "description": "Staircase-style seating for reading",
        "group_friendly": False,
        "max_group_size": 2,
    },
    "meeting_nook": {
        "name": "Meeting Nook",
        "capacity": 8,
        "type": "study",
        "description": "Small group area - bottom left",
        "group_friendly": True,
        "max_group_size": 8,
    },
    "carrels_top": {
        "name": "Study Carrels (Top)",
        "capacity": 18,
        "type": "individual",
        "description": "Solo desks with dividers - top edge",
        "group_friendly": False,
        "max_group_size": 1,
    },
    "carrels_bottom": {
        "name": "Study Carrels (Bottom)",
        "capacity": 18,
        "type": "individual",
        "description": "Solo desks with dividers - bottom edge",
        "group_friendly": False,
        "max_group_size": 1,
    },
    "multimedia_room": {
        "name": "Multimedia Room",
        "capacity": 12,
        "type": "room",
        "description": "Multimedia facilities - right side",
        "group_friendly": True,
        "max_group_size": 6,
    },
    "discussion_room": {
        "name": "Discussion Room",
        "capacity": 8,
        "type": "room",
        "description": "Requires advance booking - admin controlled",
        "group_friendly": True,
        "max_group_size": 8,
        "bookable": True,
    },
}

# ============================================================
# DISCUSSION ROOM BOOKINGS (Queue-based)
# Supports multiple bookings throughout the day.
# Each entry: {booked_by, start_time, end_time, purpose}
# ============================================================

DEFAULT_BOOKINGS = []


def _generate_default_bookings():
    """
    Makes sample bookings based on the current time so the demo always
    shows realistic upcoming bookings. It never makes a booking that is
    happening right now, so we can still test the real booking flow.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    
    def _round_to_next_30(dt):
        """Round dt forward to the next :00 or :30 mark."""
        if dt.minute < 30:
            return dt.replace(minute=30, second=0, microsecond=0)
        else:
            return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    
    # Booking 1: starts 1 hour from now (TODAY, future)
    start1_dt = _round_to_next_30(now + timedelta(minutes=60))
    end1_dt = start1_dt + timedelta(minutes=90)
    start1 = start1_dt.strftime("%I:%M %p").lstrip("0")
    end1 = end1_dt.strftime("%I:%M %p").lstrip("0")
    
    # Booking 2: starts 3.5 hours from now (TODAY, future)
    start2_dt = _round_to_next_30(now + timedelta(minutes=210))
    end2_dt = start2_dt + timedelta(minutes=90)
    start2 = start2_dt.strftime("%I:%M %p").lstrip("0")
    end2 = end2_dt.strftime("%I:%M %p").lstrip("0")
    
    # Booking 3: TOMORROW morning
    start3 = "9:00 AM"
    end3 = "10:30 AM"
    
    bookings = []
    
    # Only include today's bookings if they don't go past 9 PM
    if start1_dt.hour < 21:
        bookings.append({
            "id": "bk001",
            "date": today_str,
            "booked_by": "BSIT 3-1",
            "purpose": "Capstone meeting",
            "start_time": start1,
            "end_time": end1,
        })
    if start2_dt.hour < 21 and start2_dt.date() == now.date():
        bookings.append({
            "id": "bk002",
            "date": today_str,
            "booked_by": "BSCS 2-2",
            "purpose": "Group study",
            "start_time": start2,
            "end_time": end2,
        })
    
    bookings.append({
        "id": "bk003",
        "date": tomorrow_str,
        "booked_by": "BSIT 4-1",
        "purpose": "Thesis review",
        "start_time": start3,
        "end_time": end3,
    })
    
    return bookings


# Populated when the module is first imported
DEFAULT_BOOKINGS = _generate_default_bookings()

# Kept for backward compatibility
DISCUSSION_ROOMS = {
    "discussion_room_1f": {
        "name": "Discussion Room (1F)",
        "status": "occupied",
        "booked_by": DEFAULT_BOOKINGS[0]["booked_by"] if DEFAULT_BOOKINGS else "-",
        "free_at": DEFAULT_BOOKINGS[0]["end_time"] if DEFAULT_BOOKINGS else "-",
    },
}


def get_current_booking(bookings=None):
    """Returns the currently active booking TODAY, or None if room is free."""
    if bookings is None:
        bookings = DEFAULT_BOOKINGS
    
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time_min = now.hour * 60 + now.minute
    
    for booking in bookings:
        # Only consider today's bookings
        if booking.get("date") != today_str:
            continue
        try:
            start = _parse_time_to_minutes(booking["start_time"])
            end = _parse_time_to_minutes(booking["end_time"])
            if start <= current_time_min < end:
                return booking
        except (ValueError, KeyError):
            continue
    return None


def get_upcoming_bookings(bookings=None, limit=3):
    """Returns list of upcoming bookings today and tomorrow."""
    if bookings is None:
        bookings = DEFAULT_BOOKINGS
    
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time_min = now.hour * 60 + now.minute
    
    upcoming = []
    for booking in bookings:
        b_date = booking.get("date", today_str)
        try:
            start = _parse_time_to_minutes(booking["start_time"])
            # Today's future bookings
            if b_date == today_str and start > current_time_min:
                upcoming.append(booking)
            # Future days' bookings
            elif b_date > today_str:
                upcoming.append(booking)
        except (ValueError, KeyError):
            continue
    
    # Sort by date then time
    upcoming.sort(key=lambda b: (b.get("date", ""), _parse_time_to_minutes(b["start_time"])))
    return upcoming[:limit]


def _parse_time_to_minutes(time_str):
    """Parse '10:00 AM' or '2:30 PM' to minutes since midnight."""
    try:
        dt = datetime.strptime(time_str.strip(), "%I:%M %p")
        return dt.hour * 60 + dt.minute
    except ValueError:
        return 0


def get_next_free_time(bookings=None):
    """Returns the time when the room becomes free after the current booking."""
    current = get_current_booking(bookings)
    if current:
        return current["end_time"]
    return None


# ============================================================
# SIMULATED OCCUPANCY COUNTS
# ============================================================

# ============================================================
# LIVE CAMERA COUNTS
# If counts.json exists (written by camera_worker.py), read real counts
# from it. For any zone not in counts.json, fall back to simulated numbers.
# This way the dashboard works whether or not the camera worker is running.
# ============================================================

import json
from pathlib import Path

COUNTS_FILE = Path(__file__).parent / "counts.json"
LIVE_DATA_MAX_AGE_SECONDS = 10   # worker heartbeat considered fresh within this
LIVE_HOLD_SECONDS = 120          # how long to HOLD the last real count through a hiccup

# remember the last good live reading. when the camera worker has a
# problem (wifi drop, reconnect, restart) we keep showing the last real
# number for up to LIVE_HOLD_SECONDS instead of switching the live zone
# back to simulated data. a fake number jumping around is worse than a
# real number frozen for a bit.
_LAST_GOOD = {"counts": {}, "ts": 0.0}


def _zone_is_fresh(meta, now_dt):
    """True if this zone metadata says it updated recently and is live
    or reconnecting."""
    try:
        upd = datetime.fromisoformat(meta.get("updated_at", ""))
    except (ValueError, TypeError):
        return False
    if (now_dt - upd).total_seconds() > LIVE_DATA_MAX_AGE_SECONDS:
        return False
    return meta.get("status") in ("live", "reconnecting")


def _read_live_counts():
    """Read counts.json and return zone_id -> count.

    If the worker wrote the per zone "zones" info, we check freshness per
    zone (ready for multiple cameras). If not, we use the old top level
    status and updated_at. If everything is stale or broken we hold the
    last good reading for LIVE_HOLD_SECONDS before giving up.
    """
    now_dt = datetime.now()
    fresh = {}
    try:
        if COUNTS_FILE.exists():
            data = json.loads(COUNTS_FILE.read_text())
            counts = data.get("counts", {}) or {}
            zones_meta = data.get("zones") or {}
            if zones_meta:
                # check each zone by itself, so one camera reconnecting
                # does not hide a healthy one
                for zone_id, value in counts.items():
                    meta = zones_meta.get(zone_id, {})
                    if _zone_is_fresh(meta, now_dt):
                        fresh[zone_id] = value
            else:
                # old format (single camera)
                ok_status = data.get("status") in ("live", "reconnecting")
                try:
                    upd = datetime.fromisoformat(data.get("updated_at", ""))
                    ok_age = (now_dt - upd).total_seconds() <= LIVE_DATA_MAX_AGE_SECONDS
                except (ValueError, TypeError):
                    ok_age = False
                if ok_status and ok_age:
                    fresh = dict(counts)
    except (json.JSONDecodeError, OSError):
        fresh = {}

    if fresh:
        _LAST_GOOD["counts"] = dict(fresh)
        _LAST_GOOD["ts"] = time.time()
        return fresh

    # nothing fresh: keep the last good numbers during short outages
    if _LAST_GOOD["counts"] and (time.time() - _LAST_GOOD["ts"]) <= LIVE_HOLD_SECONDS:
        return dict(_LAST_GOOD["counts"])
    return {}


def get_current_counts():
    """
    Returns a dict of zone_id -> current people count.
    
    Zones with a camera get the real count from counts.json. The other
    zones get simulated numbers so the dashboard still looks complete
    during the prototype phase.
    """
    live_counts = _read_live_counts()

    # simulated counts for zones with no camera yet. each zone is seeded
    # by its own name so it holds a steady believable number instead of
    # jumping every refresh (that looked broken next to the real live
    # zone). the number slowly drifts every 3 minutes or so.
    drift_bucket = int(time.time() // 180)

    counts = {}
    for zone_id, zone in ZONES.items():
        if zone_id in live_counts:
            # real count from the camera (limit to capacity just in case)
            counts[zone_id] = min(live_counts[zone_id], zone["capacity"])
        else:
            cap = zone["capacity"]
            rng = random.Random(f"{zone_id}-{drift_bucket}")
            counts[zone_id] = rng.randint(int(cap * 0.2), int(cap * 0.95))

    return counts


# ============================================================
# TIERED ALERT LOGIC
# ============================================================

def get_status(count, capacity):
    if capacity == 0:
        return "N/A", "#888888"
    
    percentage = (count / capacity) * 100
    
    if percentage < 50:
        return "AVAILABLE", "#22c55e"
    elif percentage < 90:
        return "BUSY", "#f97316"
    else:
        return "FULL", "#ef4444"


def get_status_emoji(status):
    if status == "AVAILABLE":
        return "🟢"
    elif status == "BUSY":
        return "🟠"
    elif status == "FULL":
        return "🔴"
    return "⚪"


# ============================================================
# SMART SUGGESTION ENGINE
# ============================================================

def get_smart_suggestion(counts, group_size=1):
    best_zone = None
    best_percentage = 100
    
    for zone_id, zone in ZONES.items():
        if zone.get("bookable"):
            continue
        
        if group_size > 1 and (not zone["group_friendly"] or zone["max_group_size"] < group_size):
            continue
        
        count = counts.get(zone_id, 0)
        capacity = zone["capacity"]
        if capacity <= 0:
            continue
        percentage = (count / capacity) * 100
        
        available_seats = capacity - count
        if available_seats >= group_size and percentage < best_percentage:
            best_percentage = percentage
            best_zone = zone_id
    
    if best_zone:
        zone = ZONES[best_zone]
        count = counts.get(best_zone, 0)
        available = zone["capacity"] - count
        return {
            "zone_id": best_zone,
            "zone_name": zone["name"],
            "available_seats": available,
            "percentage": best_percentage,
        }
    
    return None


# ============================================================
# HISTORICAL TREND DATA
# ============================================================

def get_historical_data():
    hours = [
        "7 AM", "8 AM", "9 AM", "10 AM", "11 AM", "12 PM",
        "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM"
    ]
    occupancy = [12, 35, 68, 95, 130, 155, 140, 120, 105, 88, 65, 40, 15]
    total_capacity = sum(z["capacity"] for z in ZONES.values())
    percentages = [round((o / total_capacity) * 100) for o in occupancy]
    return hours, occupancy, percentages


# ============================================================
# PEAK HOURS ADVISORY
# ============================================================

def get_peak_advisory():
    now = datetime.now()
    hour = now.hour
    
    if 11 <= hour <= 13:
        return "⚠️ You are currently in PEAK HOURS (11 AM – 1 PM). Expect high occupancy."
    elif 9 <= hour <= 10:
        return "📈 Occupancy is rising. Peak hours start at 11 AM."
    elif 14 <= hour <= 16:
        return "📉 Past peak hours. Occupancy is gradually decreasing."
    else:
        return "✅ Low traffic period. Plenty of seats available."