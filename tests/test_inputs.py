import cv2
import numpy as np
import pytest

from triffid_uav.config import load_config
from triffid_uav.inputs import discover_flight, discover_topview, resolve_topview_frame, select_orbit_frames
from conftest import srt_block


def test_orbit_frames_are_selected_and_saved(project):
    config = load_config(None, root=project)
    flight = discover_flight(config.paths)
    sources, info, settings = select_orbit_frames(flight, config.selection, project / 'outputs')
    assert len(sources) == 8 and settings['views_per_pitch'] == 8 and info.fps == 10
    for source in sources:
        frame = cv2.imread(str(project / 'outputs' / source['image_file']))
        assert frame.mean() == pytest.approx(source['frame_index'], abs=2)
        assert source['telemetry']['lat'] is not None


def test_topview_without_image_reads_flight_frame_at_framecnt(project):
    config = load_config(None, root=project)
    source = resolve_topview_frame(discover_topview(config.paths), discover_flight(config.paths).video,
                                   project / 'outputs')
    assert source['source_type'] == 'video_frame' and source['frame_count'] == 31 and source['frame_index'] == 30
    assert cv2.imread(str(project / 'outputs/frames/topview.png')).mean() == pytest.approx(30, abs=2)
    assert source['telemetry']['gimbal_pitch'] == -90


def test_topview_image_takes_precedence(project):
    cv2.imwrite(str(project / 'input_topview/nadir.jpg'), np.full((40, 50, 3), 200, np.uint8))
    config = load_config(None, root=project)
    source = resolve_topview_frame(discover_topview(config.paths), None, project / 'outputs')
    assert source['source_type'] == 'photo' and (source['width'], source['height']) == (50, 40)
    assert source['frame_count'] == 31


def test_multi_block_topview_srt_uses_most_nadir_block(project):
    (project / 'input_topview/topview.srt').write_text('\n\n'.join(
        [srt_block(11, 1, 38, 23, 0, -60), srt_block(41, 4, 38, 23, 0, -88), srt_block(51, 5, 38, 23, 0, -70)]))
    config = load_config(None, root=project)
    source = resolve_topview_frame(discover_topview(config.paths), discover_flight(config.paths).video,
                                   project / 'outputs')
    assert source['frame_count'] == 41


def test_ambiguous_or_missing_inputs_fail_clearly(project):
    (project / 'inputs/second.mp4').write_bytes(b'x')
    with pytest.raises(ValueError, match='Expected one flight video'):
        discover_flight(load_config(None, root=project).paths)
    (project / 'input_topview/topview.srt').unlink()
    with pytest.raises(ValueError, match='No topview SRT'):
        discover_topview(load_config(None, root=project).paths)
