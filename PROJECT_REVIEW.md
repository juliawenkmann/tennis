# Project Review

This review focuses on what most affects whether the project is useful for post-game tennis analysis rather than just visually impressive.

## Main Findings

1. The project had weak temporal continuity.
Frame-by-frame detections were mostly independent, which made player positions flicker and made event extraction depend too heavily on isolated detections.

2. The ball pipeline had two realism problems.
The TrackNet input frames were stacked in reverse temporal order, and ball-center extraction relied on `HoughCircles`, which is both slower and less stable than component-based peak extraction for this heatmap output.

3. The expensive models were not runtime-aware.
YOLO and TrackNet were not using a shared device-selection path, so the project would not automatically benefit from CUDA or MPS on machines that have it.

4. The longer clips needed a repeatable stress-test path.
The project had short interactive demos, but no simple batch command to run the full pipeline on the longer samples and compare throughput and event counts.

## Improvements Implemented

1. Device-aware inference.
`src/tennis_tracker/runtime.py` now centralizes runtime selection for PyTorch and Ultralytics.

2. More stable player tracks.
`src/tennis_tracker/pipeline.py` now includes a small temporal player tracker that smooths detections, preserves near/far labels, and carries short gaps instead of dropping players immediately.

3. Better ball extraction.
`src/tennis_tracker/pipeline.py` now feeds TrackNet frames in chronological order, and `src/tennis_tracker/tracknet.py` now uses connected-component peak extraction instead of `HoughCircles`.

4. Ball trajectory filtering.
`src/tennis_tracker/pipeline.py` now rejects implausible jumps, smooths accepted ball positions, and filters out detections that project far outside the playable court region.

5. Long-clip stress runner.
`scripts/run_long_clip_suite.py` now runs the full `all` pipeline on the longer sample clips and writes a machine-readable summary.

## Still Worth Doing Next

1. Replace heuristic court detection with a learned court-keypoint model.
That is still the largest accuracy bottleneck when the camera changes or the court is partially occluded.

2. Add a tiny reviewed benchmark for the longer clips.
The reviewed benchmark is currently strong on short clips, but it needs manually checked longer-rally labels to be meaningful at project scale.

3. Separate raw detections from filtered tracks in the saved JSON.
Right now the saved tracking JSON contains the stabilized outputs only. Keeping both layers would help debugging and downstream confidence analysis.
