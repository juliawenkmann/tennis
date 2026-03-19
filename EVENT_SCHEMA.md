# Tennis Event Schema

This project already produces frame-level tracking:

- court geometry
- player detections
- ball detections

The next layer should be a clean event model on top of that tracking.

The goal is simple:

1. convert noisy frame-by-frame data into tennis events
2. export those events to XML
3. compute post-match analysis from the same event data

This file defines that event model before implementation.

## Design Rules

- Keep one canonical intermediate format: `event JSON`
- Generate XML from that JSON, not directly from tracking
- Store both raw image coordinates and derived court coordinates
- Do not force labels when confidence is low
- Only trust homography-based court coordinates for court-plane events
- Keep analysis fields derived, not hard-coded into the base schema

## What Is Reliable

Reliable now:

- court corners and court homography
- player image position
- player court position
- ball image position
- approximate rally segmentation

Reasonably derivable next:

- rally start
- rally end
- bounce
- hit/contact
- serve
- hitter side: `near_player`, `far_player`, `unknown`

Not reliable enough yet:

- forehand / backhand
- winner / forced error / unforced error
- volley / smash / slice
- score / game / set

## Canonical Output

The new intermediate file should be `match_events.json`.

Top-level structure:

```json
{
  "source_video": "broadcast_2s.mp4",
  "fps": 25.0,
  "frame_size": {"width": 640, "height": 360},
  "court_reference_m": {"width": 10.97, "length": 23.77, "singles_width": 8.23},
  "tracking_source": "out/all.json",
  "rallies": []
}
```

## Rally Schema

Each rally is one detected play segment.

```json
{
  "id": 1,
  "start_frame": 2,
  "end_frame": 48,
  "start_time_sec": 0.08,
  "end_time_sec": 1.92,
  "confidence": 0.81,
  "events": [],
  "summary": {}
}
```

Fields:

- `id`: stable rally id inside one clip
- `start_frame`, `end_frame`: inclusive frame range
- `start_time_sec`, `end_time_sec`: derived from fps
- `confidence`: confidence of the rally segmentation itself
- `events`: ordered event list
- `summary`: derived stats for fast analysis

## Event Schema

Each event should be one dictionary with the same base fields, regardless of type.

```json
{
  "id": 1,
  "type": "bounce",
  "frame": 17,
  "time_sec": 0.68,
  "actor": "far_player",
  "confidence": 0.77,
  "source": "trajectory_rule",
  "ball_image_xy": [353.0, 270.0],
  "ball_court_xy_m": [5.22, 6.81],
  "player_image_xy": [295.3, 103.4],
  "player_court_xy_m": [4.73, 1.23],
  "extra": {}
}
```

Required base fields:

- `id`
- `type`
- `frame`
- `time_sec`
- `actor`
- `confidence`
- `source`

Optional coordinate fields:

- `ball_image_xy`
- `ball_court_xy_m`
- `player_image_xy`
- `player_court_xy_m`

Free-form extension field:

- `extra`

## Allowed Event Types

Phase 1 should support only these event types:

- `rally_start`
- `serve`
- `hit`
- `bounce`
- `rally_end`

Optional later:

- `net_cross`
- `ball_out`
- `double_bounce`
- `let`

## Event Semantics

### `rally_start`

First frame where a rally becomes active.

Fields:

- `actor`: `near_player`, `far_player`, or `unknown`
- `confidence`
- `source`

### `serve`

First hit of a rally.

Fields:

- `actor`
- `confidence`
- `source`
- `ball_image_xy`
- `player_image_xy`
- `player_court_xy_m`

Notes:

- `ball_court_xy_m` may be omitted because the ball is usually airborne at serve contact

### `hit`

Racket-ball contact after the serve.

Fields:

- `actor`
- `confidence`
- `source`
- `ball_image_xy`
- `player_image_xy`
- `player_court_xy_m`

Notes:

- `ball_court_xy_m` should normally be omitted or set to `null`
- if an estimated court projection is stored, it should go under `extra.estimated_ball_court_xy_m`

### `bounce`

Ball contact with the court.

Fields:

- `actor`: player who hit the previous shot, or `unknown`
- `confidence`
- `source`
- `ball_image_xy`
- `ball_court_xy_m`

Notes:

- this is the most trustworthy court-plane ball event

### `rally_end`

Last active rally event.

Fields:

- `actor`: usually `unknown` at first
- `confidence`
- `source`
- `extra.reason`

Allowed initial reasons:

- `ball_missing`
- `clip_end`
- `ball_out_suspected`
- `double_bounce_suspected`
- `unknown`

## Summary Schema

Each rally should also contain a compact summary.

```json
{
  "event_count": 6,
  "serve_actor": "far_player",
  "hit_count": 3,
  "bounce_count": 2,
  "duration_sec": 1.84,
  "max_ball_speed_px_per_frame": 23.4
}
```

Phase 1 summary fields:

- `event_count`
- `serve_actor`
- `hit_count`
- `bounce_count`
- `duration_sec`
- `ball_detected_frame_count`

Later summary fields:

- `max_ball_speed_px_per_frame`
- `near_player_distance_m`
- `far_player_distance_m`
- `mean_recovery_to_center_m`

## Mapping From Current Tracking

Current tracking already provides most of the raw input:

- `frame_index`
- `timestamp_sec`
- `court.image_corners`
- `players[].label`
- `players[].image_xy`
- `players[].court_xy_m`
- `ball.image_xy`
- `ball.court_xy_m`

That means the event extractor only needs to add:

- temporal segmentation
- event classification
- event confidence
- actor attribution

## Minimal Detection Plan

The implementation should stay heuristic first.

### Step 1: trajectory cleanup

- smooth ball positions over time
- bridge short gaps
- reject impossible jumps

### Step 2: rally segmentation

- start a rally when ball detections become stable
- end a rally when the ball is missing for too long or the clip ends

### Step 3: bounce detection

Detect local extrema in ball court `y` and strong direction changes.

Use:

- sign change in vertical motion
- short temporal consistency window
- bounce spacing threshold

### Step 4: hit detection

Infer hit frames from:

- ball direction change
- proximity to one player
- ball height proxy in image space
- alternating hitter side constraint

### Step 5: serve detection

The first reliable hit in a rally becomes `serve`.

Heuristic:

- early in rally
- one player near baseline
- ball starts on server side

### Step 6: rally end reason

Classify minimally:

- ball track disappears after a bounce: `ball_missing`
- clip ends mid-rally: `clip_end`
- final bounce outside court bounds: `ball_out_suspected`

## XML Mapping

The XML export should be generated from the event JSON.

Phase 1 mapping:

- `match`
- `court`
- `rally`
- `frame` entries for key events only

Recommended improvement over the current notebook:

- keep compatible frame tags
- add event type from canonical event JSON
- do not treat every event as only a plain frame marker

Target XML shape:

```xml
<match>
  <court ... />
  <rally id="1">
    <frame id="2" time="0.08" event="rally_start" ... />
    <frame id="3" time="0.12" event="serve" ... />
    <frame id="17" time="0.68" event="bounce" ... />
    <frame id="22" time="0.88" event="hit" ... />
    <frame id="48" time="1.92" event="rally_end" ... />
  </rally>
</match>
```

## Analysis Outputs Enabled By This

Once the event layer exists, post-game analysis becomes straightforward.

Immediate analyses:

- serve placement map
- bounce heatmap
- rally length distribution
- player court position at hit
- player court position at bounce
- recovery distance after hit

Later analyses:

- pressure patterns
- depth of play
- shot direction tendencies
- player movement efficiency

## Recommended Implementation Order

1. add `events.py` with the canonical JSON generator
2. emit `match_events.json` from tracked clips
3. generate XML from `match_events.json`
4. create one notebook to inspect events on video and bird's-eye view
5. only then add richer tennis semantics

## Practical Success Criteria

The first version is good enough if:

- rallies are segmented reasonably
- most serves are marked correctly
- many bounce events are correct
- hit attribution is often right, even if not perfect
- XML and analysis both consume the same event JSON

That is a much better foundation than trying to infer high-level tennis labels directly from a fragile CNN output.
