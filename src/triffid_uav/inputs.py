"""Input discovery and frame resolution: inputs/ (flight) and input_topview/ (nadir capture).

inputs/         flight.mp4 + flight.srt        -> N orbit frames chosen from telemetry
input_topview/  topview.srt [+ nadir image]    -> one nadir frame:
                  * image present: that image, with the SRT block as its telemetry
                  * no image:      inputs/flight.mp4 at the SRT block's FrameCnt
Frames are written to <outputs>/frames/ and referenced relative to <outputs>.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import logging
import math

from .frame_selection import select_views
from .media import VideoReader, read_rgb, write_rgb
from .srt_metadata import SrtFrameMeta, parse_srt

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
SRT_SUFFIXES = {'.srt'}


@dataclass(frozen=True)
class FlightInputs:
    video: Path
    srt: Path


@dataclass(frozen=True)
class TopviewInputs:
    srt: Path
    image: Path | None


def _single(directory: Path, suffixes: set[str], what: str, required: bool = True) -> Path | None:
    if not directory.is_dir():
        if required:
            raise ValueError(f'Missing input folder for {what}: {directory}')
        return None
    found = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if len(found) > 1:
        raise ValueError(f'Expected one {what} in {directory}, found {[p.name for p in found]}; '
                         f'set it explicitly in the config paths section')
    if not found and required:
        raise ValueError(f'No {what} ({", ".join(sorted(suffixes))}) in {directory}')
    return found[0] if found else None


def _explicit(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_file():
        raise ValueError(f'Configured input file does not exist: {path}')
    return path


def discover_flight(paths) -> FlightInputs:
    folder = Path(paths.inputs)
    return FlightInputs(_explicit(paths.flight_video) or _single(folder, VIDEO_SUFFIXES, 'flight video'),
                        _explicit(paths.flight_srt) or _single(folder, SRT_SUFFIXES, 'flight SRT'))


def discover_topview(paths) -> TopviewInputs:
    folder = Path(paths.topview)
    return TopviewInputs(_explicit(paths.topview_srt) or _single(folder, SRT_SUFFIXES, 'topview SRT'),
                         _explicit(paths.topview_image) or _single(folder, IMAGE_SUFFIXES, 'topview image',
                                                                   required=False))


def _read_srt(path: Path) -> list[SrtFrameMeta]:
    records = parse_srt(path.read_text(encoding='utf-8-sig', errors='replace'))
    if not records:
        raise ValueError(f'No telemetry blocks in {path}')
    return records


def _source(source_id, source_type, relative_image, rgb, record: SrtFrameMeta, timestamp_s, **extra) -> dict:
    source = {'source_id': source_id, 'source_type': source_type, 'image_file': relative_image,
              'frame_index': record.frame_index, 'timestamp_s': timestamp_s,
              'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
              'telemetry': asdict(record), 'observations': []}
    source.update(extra)
    return source


def select_orbit_frames(flight: FlightInputs, selection, outputs: Path):
    """Choose orbit views from the flight SRT, save their frames, return (sources, video_info, settings)."""
    records = _read_srt(flight.srt)
    sources = []
    with VideoReader(flight.video) as reader:
        views, settings = select_views(records, reader.info.fps, reader.info.frame_count, **selection.kwargs())
        for view in views:
            rgb = reader.read_rgb(view.frame_index)
            relative = f'frames/{view.view_id}.png'
            write_rgb(outputs / relative, rgb)
            sources.append(_source(
                view.view_id, 'video_frame', relative, rgb, view.telemetry, view.timestamp_s,
                target_angle_deg=view.target_angle_deg, actual_angle_deg=view.actual_angle_deg,
                angular_error_deg=view.angular_error_deg,
                telemetry_time_delta_s=view.timestamp_s - view.telemetry.timestamp_s))
            # Keep the frame index of the decoded video frame (telemetry may be lower-rate).
            sources[-1]['frame_index'] = view.frame_index
        info = reader.info
    return sources, info, settings


def nadir_record(records: list[SrtFrameMeta]) -> SrtFrameMeta:
    """The block itself when the SRT has one; otherwise the one with pitch closest to -90 deg."""
    if len(records) == 1:
        return records[0]
    def key(r):
        pitch = r.gimbal_pitch if r.gimbal_pitch is not None and math.isfinite(r.gimbal_pitch) else math.inf
        return abs(pitch + 90), r.timestamp_s
    chosen = min(records, key=key)
    log.warning('topview SRT has %d blocks; using FrameCnt %d (pitch %s)', len(records),
                chosen.frame_index + 1, chosen.gimbal_pitch)
    return chosen


def resolve_topview_frame(topview: TopviewInputs, flight_video: Path | None, outputs: Path) -> dict:
    """Return the polygon source for the nadir frame and save its pixels to frames/topview.png."""
    record = nadir_record(_read_srt(topview.srt))
    if topview.image is not None:
        rgb = read_rgb(topview.image)
        source_type, origin = 'photo', str(topview.image)
    else:
        if flight_video is None:
            raise ValueError('No topview image and no flight video to take the nadir frame from')
        with VideoReader(flight_video) as reader:
            rgb = reader.read_rgb(record.frame_index)
        source_type, origin = 'video_frame', f'{flight_video}#frame={record.frame_index}'
    relative = 'frames/topview.png'
    write_rgb(outputs / relative, rgb)
    return _source('topview', source_type, relative, rgb, record, record.timestamp_s,
                   frame_count=record.frame_index + 1, frame_origin=origin, srt_file=str(topview.srt))
