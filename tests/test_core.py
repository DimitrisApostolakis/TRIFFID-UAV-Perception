"""Model-independent building blocks carried over from the TRIFFID UAV package."""
import math

import numpy as np
import pytest

from triffid_uav.frame_selection import angular_distance, select_views
from triffid_uav.masks import decode_rle, encode_rle, observations_from_prediction
from triffid_uav.models import BACKENDS, build_backend
from triffid_uav.models.base import ClassPrediction
from triffid_uav.ontology import BY_ID, CLASSES, classes_for
from triffid_uav.srt_metadata import SrtFrameMeta, parse_srt


def test_ontology_counts():
    assert len(CLASSES) == 38
    assert sum(c.geometry == 'Point' for c in CLASSES) == 18
    assert BY_ID[13].name == 'window' and BY_ID[14].name == 'building'
    with pytest.raises(ValueError):
        classes_for('Point', 'building')


@pytest.mark.parametrize('shape', [(1, 1), (5, 7), (12, 3)])
def test_rle_roundtrip(shape):
    rng = np.random.default_rng(0)
    mask = rng.random(shape) > .5
    assert (decode_rle(encode_rle(mask)) == mask).all()


def test_components_have_foreground_raycast_point():
    mask = np.zeros((30, 30), bool)
    mask[5:25, 5:25] = True
    mask[10:20, 10:20] = False  # hole at the centroid
    obs = observations_from_prediction(ClassPrediction(13, mask), BY_ID[13], 'v')[0]
    x, y = obs['raycast_point_xy']
    assert mask[y, x] and obs['bbox_xyxy'] == [5, 5, 25, 25]


def test_srt_parser_reads_framecnt_and_gimbal():
    text = ('7\n00:00:01,000 --> 00:00:01,033\n<font>FrameCnt: 7, DiffTime: 33ms\n'
            '[latitude: 46.1] [longitude: 1.3] [rel_alt: 86.0 abs_alt: 680.0] '
            '[gb_yaw: -43.8 gb_pitch: -58.1 gb_roll: 0.0] </font>')
    record = parse_srt(text)[0]
    assert record.frame_index == 6 and record.gimbal_pitch == -58.1 and record.lat == 46.1


def test_orbit_selection_uses_position_not_yaw():
    records = []
    for i, degrees in enumerate(range(0, 360, 45)):
        a = math.radians(degrees)
        records.append(SrtFrameMeta(i * 10, i, 38 + .001 * math.cos(a),
                                    23 + .001 * math.sin(a) / math.cos(math.radians(38)), 150, 50, 0, -45, 0))
    views, _ = select_views(records, 10, 80)
    assert [round(v.actual_angle_deg) % 360 for v in views] == list(range(0, 360, 45))
    assert angular_distance(359, 1) == 2


def test_backend_registry():
    assert 'fss-sam3' in BACKENDS

    class Config:
        backend = 'nope'
    with pytest.raises(ValueError, match='Unknown model.backend'):
        build_backend(Config())


def test_fss_sam3_canvases_for_one_and_five_shots():
    from triffid_uav.models.fss_sam3 import five_shot_canvas, one_shot_canvas
    query = np.zeros((48, 64, 3), np.uint8)
    support = np.zeros((20, 30, 3), np.uint8)
    mask = np.zeros((20, 30), np.uint8)
    mask[4:16, 5:25] = 1
    canvas, box, crop = one_shot_canvas(support, mask, query, size=500, ratio=.6)
    assert canvas.size == (500, 500) and crop == (0, 0, 500, 200) and all(0 < v <= 1 for v in box)
    canvas, boxes, crop = five_shot_canvas([(support, mask)] * 5, query, size=500, ratio=.3)
    assert len(boxes) == 5 and crop == (150, 150, 350, 350)


def test_fss_sam3_backend_validates_assets_and_options(tmp_path):
    from triffid_uav.models.fss_sam3 import FssSam3Backend
    with pytest.raises(ValueError, match='Missing FSS-SAM3 asset'):
        FssSam3Backend(tmp_path, tmp_path / 'w.pt', tmp_path / 'bpe.gz')
    with pytest.raises(ValueError, match='prompt'):
        FssSam3Backend(tmp_path, tmp_path / 'w.pt', tmp_path / 'bpe.gz', prompt='text')
