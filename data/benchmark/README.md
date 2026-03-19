# Benchmark Labels

This folder stores manually reviewed event labels for sample clips.

Label files live in `data/benchmark/labels/` and use a compact structure:

```json
{
  "source_video": "broadcast_2s.mp4",
  "annotation_status": "review_needed",
  "notes": "...",
  "rallies": [
    {
      "id": 1,
      "start_frame": 2,
      "end_frame": 49,
      "events": [
        {"type": "serve", "frame": 4, "actor": "near_player"},
        {"type": "bounce", "frame": 11, "actor": "near_player"}
      ]
    }
  ]
}
```

Important:

- new label files are seeded from the current predictions
- the seeded files are not ground truth yet
- change `annotation_status` after review

Use `tennis-event-benchmark.ipynb` to inspect the clip, edit the JSON, save it, and score the current predictions against the reviewed labels.
