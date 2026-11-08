import pytest

from triffid_uav.config import load_config


def test_defaults_resolve_paths_against_root(tmp_path):
    config = load_config(None, root=tmp_path)
    assert config.model.shots == 5 and config.model.prompt == 'visual'
    assert config.paths.inputs == str((tmp_path / 'inputs').resolve())
    assert config.selection.kwargs()['views'] == 8


def test_yaml_and_overrides(tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('model:\n  shots: 5\npoints:\n  classes: [window]\n')
    config = load_config(path, ['model.shots=1', 'selection.views=4', "polygons.classes=[building, 'green tree']"])
    assert config.model.shots == 1 and config.selection.views == 4
    assert config.points.classes == ['window'] and config.polygons.classes == ['building', 'green tree']


@pytest.mark.parametrize('override', ['model.shots=3', 'model.prompt=text', 'model.unknown=1',
                                      'nonsense.key=1', 'points.min_area_px=0'])
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(None, [override])


def test_shipped_default_config_loads():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/default.yaml')
    assert config.model.backend == 'fss-sam3' and config.points.classes == 'auto'
