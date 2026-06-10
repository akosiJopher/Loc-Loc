# LOC-LOC v2.2.0 - Deployment Guide

Real time library occupancy monitoring. A YOLO tracking worker reads the
Tapo C200 camera over RTSP and writes a stable count, then a Streamlit
dashboard shows it.

## 1. Quick start (one machine, on the library wifi)

Windows:
```
py -m pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   (then edit it)
run.bat
```

Mac / Linux:
```
python3 -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml      (then edit it)
./run.sh
```

run.bat / run.sh starts the camera worker AND the dashboard. If you want
to run them by hand:

```
py camera_worker.py                                   (window 1: detection)
py -m streamlit run app.py --server.address 0.0.0.0   (window 2: dashboard)
```

Open the Network URL on your phone (same wifi) or scan the QR code in
the dashboard.

## 2. What the system already handles by itself

- Crash recovery: run.bat / run.sh restart the worker automatically if
  it crashes (5 sec delay). If it refused to start because there are no
  credentials, it does NOT restart, it tells you to fix secrets.toml.
- Camera dropouts: RTSP uses TCP (more stable on wifi), frames are read
  on their own thread, and reconnecting is automatic. While the frames
  are stale the worker stops running detection (no point wasting CPU on
  a frozen image) and reports "reconnecting".
- Dashboard during dropouts: the dashboard keeps showing the last REAL
  count for up to 2 minutes during a worker hiccup, instead of switching
  the live zone to fake simulated numbers.
- Count speed: the count is written the moment it changes (plus a 1 sec
  heartbeat), so a person entering or leaving shows on the dashboard
  within about 2 seconds after the stability checks pass.
- Occlusion accuracy: locloc_bytetrack.yaml gives tracks around 8 sec of
  memory. A person hidden behind someone else comes back with the SAME
  id, so no count dip and no double counting.
- More cameras later: every worker only writes its own zone into
  counts.json, so you can run a second worker like this:
  `LOCLOC_ZONE=library_lounge LOCLOC_CAM_IP=192.168.x.x py camera_worker.py`
  and give it its own ROI file named roi_config.library_lounge.json
- Logs: the worker writes worker.log (rotates at 1 MB, keeps 3 files).
  counts.json also shows the real fps and reconnect total per zone.
- Security: the admin login locks for 60 sec after 5 wrong tries, and
  the password check is constant time. Camera passwords never show in
  the logs.

## 3. Before you deploy for real (checklist)

1. CHANGE THE ADMIN PASSWORD. Add this to .streamlit/secrets.toml:
   ```
   [admin]
   password = "your-strong-password"
   ```
   The built in default is for development only.
2. Put the real camera username/password and IP in secrets.toml. Also
   give the camera a DHCP reservation in the router so its IP does not
   change after a reboot.
3. Keep it LAN only. `--server.address 0.0.0.0` makes it visible to the
   local network which is the plan (phones on library wifi). Do NOT port
   forward it to the internet. If remote access is really needed someday
   it should go behind a reverse proxy with HTTPS and a login.
4. Run it as a service so it survives restarts:
   - Windows: Task Scheduler, two "At startup" tasks. One runs
     run_worker.bat, the other runs
     `py -m streamlit run app.py --server.address 0.0.0.0`
     (or use NSSM to install both as Windows services)
   - Linux: two systemd units, example for the worker:
     ```
     [Unit]
     Description=LOC-LOC camera worker
     After=network-online.target
     [Service]
     WorkingDirectory=/opt/loc-loc
     ExecStart=/usr/bin/python3 camera_worker.py
     Restart=always
     RestartSec=5
     [Install]
     WantedBy=multi-user.target
     ```
5. Tuning (in [detection] of secrets.toml):
   - person shows up faster when entering: lower confirm_frames (min 2)
   - person disappears faster when leaving: lower grace_frames (like 12)
     but the count can dip if someone is hidden longer than that
   - CPU too high: infer_imgsz = 480, or model_path = "yolo26n.pt", or
     lower target_fps. You can see the real fps inside counts.json.
