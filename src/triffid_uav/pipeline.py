"""Orchestration: inputs -> frames -> FSS-SAM3 (one model load) -> outputs/*.json.

Points run on the orbit frames, polygons on the nadir frame. All parameters come
from `Config`; the backend is injectable for tests.
"""
from __future__ import annotations

from pathlib import Path
import logging
import time

from .config import Config
from .export import SCHEMA_VERSION, document, write_json
from .inputs import discover_flight, discover_topview, resolve_topview_frame, select_orbit_frames
from .masks import observations_from_prediction
from .media import read_rgb
from .models import build_backend
from .ontology import classes_for
from .supports import SupportBank

log = logging.getLogger(__name__)

POINTS_FILE = 'point_detections.json'
POLYGONS_FILE = 'polygon_detections.json'


def task_classes(task, geometry: str, bank: SupportBank, shots: int):
    """Configured classes for one task; 'auto' = every class of that geometry with `shots` valid supports."""
    if not task.enabled:
        return ()
    if task.classes == 'auto':
        classes = bank.ready_classes(geometry, shots)
        if not classes:
            raise ValueError(f'No {geometry} class has {shots} valid supports in {bank.path}')
        return classes
    names = task.classes if isinstance(task.classes, list) else [task.classes]
    return classes_for(geometry, ','.join(str(n) for n in names))


def infer_source(backend, rgb, classes, source: dict, min_area_px: int) -> None:
    """Run every class on one frame and append its connected components to source['observations']."""
    expected = {c.id: c for c in classes}
    seen = set()
    for prediction in backend.predict(rgb, classes):
        if prediction.class_id not in expected or prediction.class_id in seen:
            raise ValueError('Backend returned an unexpected or repeated class')
        seen.add(prediction.class_id)
        if prediction.mask.shape != rgb.shape[:2]:
            raise ValueError('Backend mask dimensions must match the original input image')
        source['observations'].extend(observations_from_prediction(
            prediction, expected[prediction.class_id], source['source_id'], min_area_px=min_area_px))
    if seen != set(expected):
        raise ValueError('Backend must return a prediction per requested class, including empty masks')
    log.info('%s: %d observations', source['source_id'], len(source['observations']))


def _infer_all(backend, sources, classes, outputs: Path, min_area_px: int) -> None:
    for source in sources:
        infer_source(backend, read_rgb(outputs / source['image_file']), classes, source, min_area_px)


def run(config: Config, backend=None) -> dict:
    """Execute the full pipeline; returns the run summary written to outputs/run.json."""
    started_run = time.perf_counter()
    timings = {}
    outputs = Path(config.paths.outputs)
    outputs.mkdir(parents=True, exist_ok=True)

    flight = discover_flight(config.paths) if config.points.enabled else None
    topview = discover_topview(config.paths) if config.polygons.enabled else None
    if topview is not None and topview.image is None and flight is None:
        flight = discover_flight(config.paths)  # nadir pixels come from the flight video

    bank = SupportBank(config.paths.supports)
    shots = config.model.shots
    point_classes = task_classes(config.points, 'Point', bank, shots)
    polygon_classes = task_classes(config.polygons, 'Polygon', bank, shots)
    supports = bank.select(point_classes + polygon_classes, shots=shots)

    started = time.perf_counter()
    orbit_sources, video_info, selection = ([], None, None)
    if config.points.enabled:
        orbit_sources, video_info, selection = select_orbit_frames(flight, config.selection, outputs)
    topview_source = resolve_topview_frame(topview, flight.video if flight else None, outputs) \
        if config.polygons.enabled else None
    timings['frames_s'] = time.perf_counter() - started

    started = time.perf_counter()
    backend = backend or build_backend(config.model)
    backend.prepare(point_classes + polygon_classes, supports)
    timings['model_preparation_s'] = time.perf_counter() - started

    common = {'support_bank': supports.metadata(), 'image_files_relative_to': 'outputs directory'}
    written = {}
    if config.points.enabled:
        started = time.perf_counter()
        _infer_all(backend, orbit_sources, point_classes, outputs, config.points.min_area_px)
        payload = document('point_detections', orbit_sources, backend.metadata(), point_classes)
        payload.update(common, video=video_info.to_dict(), srt_file=str(flight.srt), selection=selection,
                       postprocessing={'min_area_px': config.points.min_area_px, 'connectivity': 8})
        written['points'] = write_json(outputs / POINTS_FILE, payload)
        timings['points_s'] = time.perf_counter() - started

    if config.polygons.enabled:
        started = time.perf_counter()
        _infer_all(backend, [topview_source], polygon_classes, outputs, config.polygons.min_area_px)
        # Task name stays 'polygon_segmentation' (the observation schema consumed downstream).
        payload = document('polygon_segmentation', [topview_source], backend.metadata(), polygon_classes)
        payload.update(common, srt_file=topview_source['srt_file'],
                       postprocessing={'min_area_px': config.polygons.min_area_px, 'connectivity': 8})
        written['polygons'] = write_json(outputs / POLYGONS_FILE, payload)
        timings['polygons_s'] = time.perf_counter() - started

    timings['total_s'] = time.perf_counter() - started_run
    summary = {
        'schema_version': SCHEMA_VERSION,
        'inputs': {'flight_video': str(flight.video) if flight else None,
                   'flight_srt': str(flight.srt) if flight else None,
                   'topview_srt': str(topview.srt) if topview else None,
                   'topview_image': str(topview.image) if topview and topview.image else None},
        'outputs': {key: str(path) for key, path in written.items()},
        'point_classes': [c.name for c in point_classes],
        'polygon_classes': [c.name for c in polygon_classes],
        'counts': {
            'points_per_frame': {s['source_id']: len(s['observations']) for s in orbit_sources},
            'polygon_components': len(topview_source['observations']) if topview_source else 0,
        },
        'topview_frame_count': topview_source['frame_count'] if topview_source else None,
        'timings': timings,
        'config': config.to_dict(),
    }
    write_json(outputs / 'run.json', summary)
    return summary


def check(config: Config) -> dict:
    """Validate inputs, supports and classes and extract the frames, without loading a model."""
    outputs = Path(config.paths.outputs)
    bank = SupportBank(config.paths.supports)
    point_classes = task_classes(config.points, 'Point', bank, config.model.shots)
    polygon_classes = task_classes(config.polygons, 'Polygon', bank, config.model.shots)
    bank.select(point_classes + polygon_classes, shots=config.model.shots)  # raises on invalid supports
    report = {'point_classes': [c.name for c in point_classes],
              'polygon_classes': [c.name for c in polygon_classes], 'shots': config.model.shots}
    flight = None
    if config.points.enabled:
        flight = discover_flight(config.paths)
        sources, info, _ = select_orbit_frames(flight, config.selection, outputs)
        report['orbit_frames'] = [{'source_id': s['source_id'], 'frame_index': s['frame_index'],
                                   'angle_deg': round(s['actual_angle_deg'], 1)} for s in sources]
        report['video'] = info.to_dict()
    if config.polygons.enabled:
        topview = discover_topview(config.paths)
        if topview.image is None and flight is None:
            flight = discover_flight(config.paths)
        source = resolve_topview_frame(topview, flight.video if flight else None, outputs)
        report['topview'] = {'frame_count': source['frame_count'], 'source_type': source['source_type'],
                             'gimbal_pitch': source['telemetry']['gimbal_pitch'], 'origin': source['frame_origin']}
    return report
