# LOC-LOC 📍

**Real-Time Library Occupancy Monitoring System**

LOC-LOC is our thesis project at LPU Manila. It shows students how full
each area of the library is in real time, so you can find a seat without
walking around the whole floor first.

Made by group **LOKASYON** (Alis, Carvajal, Dela Cruz, Li, Lumangcas).

## What it does

- Live dashboard of all 10 zones on Floor 1 with seat counts and a
  color status (green = available, orange = busy, red = full)
- One zone (Book Common Area 2) is counted by a REAL camera. A Tapo
  C200 streams over RTSP to a YOLO tracking worker that counts the
  people inside the activity zone. The other zones use simulated data
  for now (prototype phase)
- "Find a spot" search: enter your group size and it suggests the best
  area for you
- Interactive floor map. Tap any card and a bouncing arrow shows you
  where that area is on the map
- Discussion Room booking schedule
- Staff panel (password protected) for bookings, the live camera feed,
  and drawing the activity zone on the camera view
- Occupancy trend chart for the day
- Works on phone too, just scan the QR / open the network URL on the
  same wifi
- Dark mode

## How the counting works

We dont just count the boxes YOLO detects every frame, because that
number flickers when a person moves or gets blocked by someone. Instead
we use tracking (ByteTrack), so every person keeps an ID. A new ID has
to appear for a few frames before it counts, and a counted person that
disappears for a moment is kept for a short grace period. So the count
only goes up when someone really arrives and only goes down when someone
really leaves.

## Tech used

- Python 3.13
- Streamlit (dashboard)
- Ultralytics YOLO26 + ByteTrack (detection and tracking)
- OpenCV (RTSP camera reading)
- Plotly (trend chart)

## How to run

See [HOW_TO_RUN.txt](HOW_TO_RUN.txt) for the quick version or
[README_DEPLOY.md](README_DEPLOY.md) for the full deployment guide.

Short version:

```
py -m pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   (then edit it)
run.bat
```

The dashboard also works without the camera (it falls back to simulated
data), so you can try it even with no Tapo C200.

## Tests

```
py test_counting.py     (counting logic)
py test_pipeline.py     (worker to dashboard connection)
```

## Screenshots

*(coming soon)*
