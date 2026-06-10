"""Tests for counting.py. Checks the count stays stable when people move."""
from counting import (
    PresenceTracker, FallbackSmoother,
    box_in_zones, point_in_zones, ids_inside_zones, load_roi_zones,
)

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


print("\n[1] Phantom detection (1-2 frame false positive) never counts")
t = PresenceTracker(confirm_frames=3, grace_frames=15)
# A spurious id 99 appears for only 2 frames, then vanishes forever.
seq = [[99], [99], [], [], [], []]
counts = [t.update(ids) for ids in seq]
check("phantom never reaches the count", max(counts) == 0)

print("\n[2] A real person is counted after confirm_frames and stays counted")
t = PresenceTracker(confirm_frames=3, grace_frames=15)
counts = [t.update([1]) for _ in range(10)]
check("not counted before confirm (frame 1)", counts[0] == 0)
check("not counted before confirm (frame 2)", counts[1] == 0)
check("counted at confirm frame (frame 3)", counts[2] == 1)
check("stays counted afterwards", counts[-1] == 1)

print("\n[3] Person MOVES -> YOLO briefly loses them -> count does NOT dip")
t = PresenceTracker(confirm_frames=3, grace_frames=15)
for _ in range(5):          # establish & confirm the person
    t.update([1])
# Now the person turns / is occluded for 10 frames (YOLO sees nothing).
dip = []
for _ in range(10):
    dip.append(t.update([]))
check("count holds at 1 through a 10-frame occlusion", all(c == 1 for c in dip))
# they reappear (same track id), still 1, no double count
check("no double count when they reappear", t.update([1]) == 1)

print("\n[4] Person genuinely LEAVES -> count drops after the grace window")
t = PresenceTracker(confirm_frames=3, grace_frames=15)
for _ in range(5):
    t.update([1])
after = [t.update([]) for _ in range(20)]
check("still counted inside grace window (frame 15 of absence)", after[14] == 1)
check("dropped once grace is exceeded (frame 16+)", after[16] == 0)

print("\n[5] Two people milling around, swapping positions -> steady count of 2")
t = PresenceTracker(confirm_frames=3, grace_frames=15)
import random
random.seed(0)
out = []
for _ in range(5):
    t.update([1, 2])              # both confirmed
for _ in range(40):
    # Each frame, randomly drop one of them for a frame (occlusion as they move).
    present = [1, 2]
    if random.random() < 0.4:
        present.remove(random.choice([1, 2]))
    out.append(t.update(present))
check("count stays exactly 2 the whole time", all(c == 2 for c in out))

print("\n[6] Realistic arrival/departure timeline produces clean steps")
t = PresenceTracker(confirm_frames=3, grace_frames=10)
timeline = []
# 5 frames empty
for _ in range(5): timeline.append(t.update([]))
# person A arrives and stays
for _ in range(10): timeline.append(t.update([1]))
# person B arrives, both stay
for _ in range(10): timeline.append(t.update([1, 2]))
# A leaves for good, B stays
for _ in range(15): timeline.append(t.update([2]))
check("starts at 0", timeline[0] == 0)
check("reaches 1 after A confirmed", 1 in timeline[5:12])
check("reaches 2 after B confirmed", 2 in timeline[15:22])
check("settles back to 1 after A leaves", timeline[-1] == 1)
# Verify it never overshoots.
check("never exceeds 2", max(timeline) == 2)

print("\n[7] ROI geometry: head/body point-in-zone")
zones = [(0, 0, 100, 100)]
check("box fully inside counts", box_in_zones((10, 10, 30, 60), zones))
check("box fully outside does not", not box_in_zones((200, 200, 260, 280), zones))
# Seated person: body center below the table (outside) but head inside.
check("seated person counted via head", box_in_zones((40, 80, 60, 160), [(0, 0, 100, 90)]))
check("no zones -> everything counts", box_in_zones((999, 999, 1000, 1000), []))

print("\n[8] ids_inside_zones filters correctly and skips None ids")
zones = [(0, 0, 100, 100)]
boxes = [(10, 10, 30, 60), (200, 200, 260, 280), (40, 40, 60, 80)]
ids = [1, 2, None]
inside_ids, raw = ids_inside_zones(boxes, ids, zones)
check("only inside boxes counted (raw=2)", raw == 2)
check("only id 1 returned (id2 outside, id3 None)", inside_ids == [1])

print("\n[9] FallbackSmoother kills single-frame spikes")
s = FallbackSmoother(window=5)
vals = [s.update(v) for v in [3, 3, 9, 3, 3]]   # one spike to 9
check("median ignores the lone spike", vals[-1] == 3)

print(f"\n{'='*48}\n  RESULT: {passed} passed, {failed} failed\n{'='*48}")
exit(1 if failed else 0)
