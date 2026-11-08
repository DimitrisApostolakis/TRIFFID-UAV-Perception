# UAV detection JSON format

The pipeline exports image-space observations in original-image pixels, not
longitude/latitude geometry. Geographic projection is a separate downstream step
that uses each source's telemetry (frame index, GPS, gimbal) and a scene model.

## Files

| File | Contents |
|---|---|
| `outputs/point_detections.json` | `task: point_detections`; one source per orbit frame (8 by default), each with its `observations[]`; repeated objects across frames remain |
| `outputs/polygon_detections.json` | `task: polygon_segmentation`; one `topview` source (the nadir frame) with polygon observations |
| `outputs/frames/` | The exact pixels segmented; `image_file` paths are relative to `outputs/` |
| `outputs/run.json` | Inputs, classes, per-frame counts, timings and the full resolved config |

## Envelope

- `schema_version`: `triffid.uav.observations.v1`.
- `coordinate_space`: `original_image_pixels`; top-left origin, x right, y down.
- `bbox_format`: `xyxy_max_exclusive`; `[x1,y1,x2,y2]`, with width `x2-x1`.
- `class_catalogue`: original numeric class IDs, names, target geometry and source IDs.
- `model`: checkpoint, source revision, prompting, support counts, thresholds and inference settings.
- `support_bank`: manifest hash and selected image/mask hashes, shot count and example IDs.
- `sources`: frame/image records, including empty frames.
- `video`, `srt_file`, `selection`: capture and view-selection provenance (points); `srt_file` is the topview SRT for polygons.
- `deduplication`: repeats across frames are retained; every observation is one per-frame sighting.

## Per source frame

`source_id` is a stable view label, not an object ID. `image_file` identifies the
original RGB pixels; `frame_index`, `timestamp_s`, `width` and `height` bind the
observations to that frame. `telemetry` retains capture GPS/altitude and gimbal
pose. `observations` is an array and can be empty. The `topview` source also records
`frame_count` (DJI FrameCnt), `frame_origin` (image file or flight-video frame) and `srt_file`.

## Per observation

| Field | Meaning |
|---|---|
| `observation_id` | Unique sighting ID: frame + class + component; not a persistent identity |
| `class_id`, `class_name`, `source_class_ids` | Ontology label; e.g. window is 13, furniture is 50 |
| `target_geometry` | `Point` or `Polygon`; the geographic output type after projection |
| `bbox_xyxy` | Original-resolution component bounds, maximum exclusive |
| `centroid_xy` | Mean foreground pixel location; may fall outside a concave shape or inside a hole |
| `raycast_point_xy` | Actual foreground pixel nearest the centroid; use for Point projection |
| `area_px` | Number of foreground pixels in this component |
| `mask` | Exact cropped binary mask, including holes, with its original-image offset |
| `score` | Mean foreground mask probability, uncalibrated; can be null for binary-only backends |
| `score_type` | Interpretation of the score; not a guarantee of correct class or instance identity |

Mask `counts` alternate background and foreground run lengths, starting with
background. They flatten the crop **row-major** (x changes fastest), unlike COCO
column-major RLE. `size_hw` is `[height,width]`; `offset_xy` places the crop in
the full source image. Counts sum to `height*width`. Different class masks may
overlap. Connected components can merge touching windows or split one window.

## Small valid example

This example uses a deliberately simple 4x4 square to explain the encoding. Its
box/mask/score are illustrative, not a model prediction. Real files also include
complete model and support provenance.

```json
{
  "schema_version": "triffid.uav.observations.v1",
  "task": "point_detections",
  "coordinate_space": "original_image_pixels",
  "pixel_origin": "top_left",
  "axes": {
    "x": "right",
    "y": "down"
  },
  "bbox_format": "xyxy_max_exclusive",
  "class_catalogue": [
    {
      "id": 13,
      "name": "window",
      "target_geometry": "Point",
      "source_class_ids": [
        13
      ]
    }
  ],
  "model": {
    "name": "fss-sam3",
    "supports_per_class": {
      "13": 5
    },
    "prompting": "visual positive support box; no class-name text; native dummy text visual"
  },
  "sources": [
    {
      "source_id": "view_03",
      "source_type": "video_frame",
      "image_file": "frames/view_03.png",
      "frame_index": 503,
      "timestamp_s": 16.783433333333335,
      "width": 1920,
      "height": 1080,
      "telemetry": {
        "frame_index": 503,
        "timestamp_s": 16.782,
        "lat": 46.025565,
        "lon": 1.38275,
        "abs_alt": 680.067,
        "rel_alt": 86.1,
        "gimbal_yaw": -43.8,
        "gimbal_pitch": -58.1,
        "gimbal_roll": 0.0
      },
      "observations": [
        {
          "observation_id": "view_03:class_13:component_1",
          "class_id": 13,
          "class_name": "window",
          "source_class_ids": [
            13
          ],
          "target_geometry": "Point",
          "bbox_xyxy": [
            100,
            200,
            104,
            204
          ],
          "centroid_xy": [
            101.5,
            201.5
          ],
          "raycast_point_xy": [
            101,
            201
          ],
          "area_px": 16,
          "score": 0.93,
          "score_type": "mean_foreground_probability_uncalibrated",
          "mask": {
            "encoding": "binary_rle_row_major",
            "size_hw": [
              4,
              4
            ],
            "counts": [
              0,
              16
            ],
            "offset_xy": [
              100,
              200
            ]
          }
        }
      ]
    }
  ],
  "deduplication": "none; observation IDs are not persistent world-object IDs"
}
```

