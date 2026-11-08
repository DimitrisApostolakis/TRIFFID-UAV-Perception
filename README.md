# TRIFFID UAV Perception

Few-shot detection on DJI drone flights with [FSS-SAM3](https://github.com/WongKinYiu/FSS-SAM3):

- **Points** (windows, people, vehicles, furniture, …) on **8 orbit frames** selected from the flight telemetry.
- **Polygons** (trees, buildings, roads, grass, …) on **one nadir (top-view) frame**.

Each class is prompted with 1 or 5 example image/mask pairs (your choice). The output is JSON in
original image pixels, with the telemetry of every frame, ready for geographic projection downstream.

```
inputs/           flight.mp4 + flight.srt  ──► 8 orbit frames ──┐
input_topview/    topview.srt [+ image]    ──► nadir frame ─────┤
supports/         bank.json + examples     ─────────────────────┤
                                                                ▼
                                       FSS-SAM3 (one model load, 1- or 5-shot)
                                       points on orbit frames · polygons on nadir
                                                                ▼
outputs/          point_detections.json · polygon_detections.json · frames/ · run.json
```

## Folders

| Folder | You provide | Notes |
|---|---|---|
| `inputs/` | one video (`flight.mp4`) + its DJI SRT (`flight.srt`) | Orbit frames are chosen from GPS bearing about the circle centre |
| `input_topview/` | `topview.srt` (+ optional nadir image) | Mini SRT naming the nadir FrameCnt and drone position. With an image, its pixels are used; otherwise the frame is read from `inputs/flight.mp4` at that FrameCnt |
| `supports/` | support bank (`bank.json` + images/masks) | Not in the repo; mount it. See [supports/README.md](supports/README.md) |
| `outputs/` | – | Written by the run |
| model weights | `sam3.pt`, `bpe_simple_vocab_16e6.txt.gz` | Mounted at `/models` (Docker) |

Everything in these folders is git-ignored.

## Run with Docker (GPU)

Requires Docker with the NVIDIA Container Toolkit.

```bash
docker compose build
# weights folder with sam3.pt + bpe_simple_vocab_16e6.txt.gz; support bank folder with bank.json
export MODELS_DIR=/path/to/FSS-SAM3-checkpoints SUPPORTS_DIR=/path/to/support_bank

docker compose run --rm uav check                 # validate inputs/classes, extract frames, no model
docker compose run --rm uav                       # full run, configs/default.yaml (5-shot)
docker compose run --rm uav run --shots 1         # 1-shot
docker compose run --rm uav run --set points.classes='[window, citizen]' --set polygons.enabled=false
docker compose run --rm uav overlays outputs/point_detections.json   # debug overlays
```

The image pins FSS-SAM3 to commit `b82ba83`. Weights and supports are never baked into the image.

## Run locally

```bash
pip install -r requirements-dev.txt        # plus PyTorch with CUDA, and a FSS-SAM3 checkout
python -m triffid_uav run \
  --set model.repo=/path/to/FSS-SAM3 \
  --set model.checkpoint=/path/to/sam3.pt --set model.bpe=/path/to/bpe_simple_vocab_16e6.txt.gz
```

## Configuration

All parameters live in [`configs/default.yaml`](configs/default.yaml). Override any of them with
`--set section.key=value` (YAML value syntax) or pass another file with `--config`.

| Key | Default | Meaning |
|---|---|---|
| `model.shots` | `5` | Supports per class: `1` or `5` (`--shots` shortcut) |
| `model.prompt` | `visual` | `visual`, or `text+visual` to add the class name as a text prompt |
| `model.threshold` | `0.5` | Object acceptance threshold |
| `model.canvas_size`, `support_ratio_1shot`, `support_ratio_5shot` | `1008`, `0.6`, `0.3` | FSS collage layout |
| `model.device` | `cuda` | Torch device |
| `points.classes`, `polygons.classes` | `auto` | `auto` = every class of that geometry with enough valid supports, or a list of names/IDs |
| `points.enabled`, `polygons.enabled` | `true` | Run or skip a task |
| `*.min_area_px` | `16` | Smallest connected component kept |
| `selection.views` | `8` | Orbit frames for points |
| `selection.mode` | `orbit` | `orbit` (drone bearing about the circle centre) or `heading` (gimbal yaw) |
| `selection.start_s`, `end_s` | `0`, `null` | Time window holding one circle |
| `selection.orbit_centre` | `null` | `[lat, lon]`; default: centre of the GPS bounding box |
| `paths.*` | repo folders | Override folders or point at specific files |

## Outputs

| File | Contents |
|---|---|
| `point_detections.json` | One source per orbit frame with its telemetry; every connected component per class: exact cropped RLE mask, bbox, centroid, a foreground raycast pixel, area, mean mask probability |
| `polygon_detections.json` | One `topview` source (nadir frame, FrameCnt, telemetry) with polygon components in the same format |
| `frames/` | The segmented frames (`view_00.png` … `topview.png`) |
| `run.json` | Inputs, classes, per-frame counts, timings and the full resolved config |

Schema `triffid.uav.observations.v1`: see [docs/JSON_FORMAT.md](docs/JSON_FORMAT.md).
Observations are per-frame sightings; the same object seen in several orbit frames appears in
each of them.

## Code layout

```
src/triffid_uav/
  cli.py              commands: run, check, bank, overlays
  config.py           typed config (YAML + --set overrides)
  pipeline.py         orchestration: frames -> one model load -> both tasks -> outputs
  inputs.py           folder discovery, orbit frame selection, nadir frame resolution
  frame_selection.py  angular orbit/heading view selection from SRT telemetry
  srt_metadata.py     DJI SRT parser (FrameCnt, GPS, altitude, gimbal)
  supports.py         support bank: validation, 1/5-shot selection, tools
  models/             backend registry; fss_sam3.py (collage, prompts, mask restore)
  masks.py            connected components and row-major RLE
  export.py           versioned JSON, atomic writes
  ontology.py         38 TRIFFID classes (18 Point, 20 Polygon) with original IDs
  visualize.py        debug overlays
```

To add a backend, implement `prepare(classes, supports)`, `predict(rgb, classes)` and `metadata()`
(see `models/base.py`) and register a factory in `models/__init__.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a synthetic flight, top view and support bank with an injected backend, so they need
no GPU or weights.

## Third-party

FSS-SAM3 and SAM3 are fetched at build time and are subject to their own licences (the SAM License).
Their source code and weights are not redistributed here.
