import json

import pytest

from triffid_uav.config import load_config
from triffid_uav.masks import decode_rle
from triffid_uav.pipeline import check, run


def test_run_writes_points_for_orbit_and_polygons_for_nadir(project, fake_backend):
    config = load_config(None, ['points.classes=[window, citizen]', 'polygons.classes=[building]'], root=project)
    summary = run(config, backend=fake_backend)
    outputs = project / 'outputs'
    points = json.loads((outputs / 'point_detections.json').read_text())
    polygons = json.loads((outputs / 'polygon_detections.json').read_text())
    assert fake_backend.prepared == 1 and fake_backend.calls == 9  # one model load; 8 orbit + 1 nadir frame
    assert points['task'] == 'point_detections' and len(points['sources']) == 8
    assert all(len(s['observations']) == 2 for s in points['sources'])
    assert {o['target_geometry'] for s in points['sources'] for o in s['observations']} == {'Point'}
    assert polygons['task'] == 'polygon_segmentation' and len(polygons['sources']) == 1
    source = polygons['sources'][0]
    assert source['source_id'] == 'topview' and source['frame_count'] == 31
    assert source['observations'][0]['target_geometry'] == 'Polygon'
    assert decode_rle(source['observations'][0]['mask']).sum() == 30
    assert polygons['srt_file'].endswith('topview.srt') and points['srt_file'].endswith('flight.srt')
    assert (outputs / 'frames/topview.png').is_file() and len(list((outputs / 'frames').glob('view_*.png'))) == 8
    assert summary['counts']['polygon_components'] == 1
    assert json.loads((outputs / 'run.json').read_text())['config']['model']['shots'] == 5


@pytest.mark.parametrize('shots', [1, 5])
def test_shots_parameter_controls_support_selection(project, fake_backend, shots):
    config = load_config(None, [f'model.shots={shots}', 'points.classes=[window]', 'polygons.enabled=false'],
                         root=project)
    run(config, backend=fake_backend)
    points = json.loads((project / 'outputs/point_detections.json').read_text())
    assert fake_backend.shots == shots and points['support_bank']['shots'] == shots
    assert len(points['support_bank']['classes']['window']) == shots
    assert not (project / 'outputs/polygon_detections.json').exists()


def test_auto_classes_use_every_class_with_supports(project, fake_backend):
    run(load_config(None, ['polygons.enabled=false'], root=project), backend=fake_backend)
    points = json.loads((project / 'outputs/point_detections.json').read_text())
    assert len(points['class_catalogue']) == 18


def test_polygons_only_take_nadir_from_flight_video(project, fake_backend):
    config = load_config(None, ['points.enabled=false', 'polygons.classes=[building]'], root=project)
    summary = run(config, backend=fake_backend)
    assert fake_backend.calls == 1 and summary['topview_frame_count'] == 31


def test_wrong_geometry_class_is_rejected(project, fake_backend):
    with pytest.raises(ValueError):
        run(load_config(None, ['points.classes=[building]'], root=project), backend=fake_backend)


def test_check_needs_no_model(project):
    report = check(load_config(None, ['points.classes=[window]', 'polygons.classes=[building]'], root=project))
    assert len(report['orbit_frames']) == 8 and report['topview']['frame_count'] == 31
    assert report['point_classes'] == ['window']
