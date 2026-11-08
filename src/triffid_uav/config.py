"""Run configuration: typed sections loaded from YAML, overridable from the command line.

Every tunable lives here; modules receive the section they need instead of reading globals.
Override any value with ``--set section.key=value`` (YAML syntax for the value).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    inputs: str = 'inputs'                # flight video + flight SRT
    topview: str = 'input_topview'        # topview SRT (+ optional nadir image)
    outputs: str = 'outputs'
    supports: str = 'supports'            # support bank directory containing bank.json
    flight_video: str | None = None       # default: the single video file in `inputs`
    flight_srt: str | None = None         # default: the single .srt file in `inputs`
    topview_srt: str | None = None        # default: the single .srt file in `topview`
    topview_image: str | None = None      # default: the single image in `topview`, if any


@dataclass
class ModelConfig:
    backend: str = 'fss-sam3'
    shots: int = 5                        # supports per class: 1 or 5
    prompt: str = 'visual'                # 'visual' or 'text+visual'
    threshold: float = .5                 # native object acceptance threshold
    device: str = 'cuda'
    canvas_size: int = 1008
    support_ratio_1shot: float = .6
    support_ratio_5shot: float = .3
    repo: str = '/opt/fss-sam3'
    checkpoint: str = '/models/sam3.pt'
    bpe: str = '/models/bpe_simple_vocab_16e6.txt.gz'


@dataclass
class SelectionConfig:
    mode: str = 'orbit'                   # 'orbit' (GPS bearing about centre) or 'heading' (gimbal yaw)
    views: int = 8
    start_s: float = 0.0
    end_s: float | None = None
    angle_offset_deg: float = 0.0
    angle_tolerance_deg: float = 12.0
    orbit_centre: list[float] | None = None   # [lat, lon]; default: GPS bounding-box centre
    pitches: list[float] | None = None
    pitch_tolerance_deg: float = 10.0

    def kwargs(self) -> dict:
        return dict(views=self.views, mode=self.mode, start_s=self.start_s, end_s=self.end_s,
                    angle_offset=self.angle_offset_deg, tolerance=self.angle_tolerance_deg,
                    centre=tuple(self.orbit_centre) if self.orbit_centre else None,
                    pitches=tuple(self.pitches) if self.pitches else None,
                    pitch_tolerance=self.pitch_tolerance_deg)


@dataclass
class TaskConfig:
    enabled: bool = True
    classes: str | list[str] = 'auto'     # 'auto' = every class of this geometry with enough supports
    min_area_px: int = 16


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    points: TaskConfig = field(default_factory=TaskConfig)
    polygons: TaskConfig = field(default_factory=TaskConfig)

    def validate(self) -> 'Config':
        if self.model.shots not in (1, 5):
            raise ValueError('model.shots must be 1 or 5')
        if self.model.prompt not in ('visual', 'text+visual'):
            raise ValueError("model.prompt must be 'visual' or 'text+visual'")
        if self.selection.views < 1:
            raise ValueError('selection.views must be positive')
        for name in ('points', 'polygons'):
            task = getattr(self, name)
            if task.min_area_px < 1:
                raise ValueError(f'{name}.min_area_px must be positive')
        if not (self.points.enabled or self.polygons.enabled):
            raise ValueError('Enable at least one of points / polygons')
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict | None, where: str):
    values = dict(values or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f'Unknown config key(s) in {where or "root"}: {sorted(unknown)}')
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        kwargs[name] = _build(type(default), value, f'{where}.{name}'.strip('.')) if is_dataclass(default) else value
    return cls(**kwargs)


def _apply_override(raw: dict, assignment: str) -> None:
    if '=' not in assignment:
        raise ValueError(f'Override must be section.key=value, got {assignment!r}')
    path, value = assignment.split('=', 1)
    keys = path.strip().split('.')
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f'Cannot override inside non-section {path!r}')
    node[keys[-1]] = yaml.safe_load(value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                root: str | Path | None = None) -> Config:
    """Defaults <- YAML file <- overrides. Relative paths resolve against `root` (default: CWD)."""
    raw: dict[str, Any] = {}
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'Config file must contain a mapping: {path}')
        raw = loaded
    for assignment in overrides or ():
        _apply_override(raw, assignment)
    config = _build(Config, raw, '').validate()
    base = Path(root or Path.cwd()).resolve()
    for f in fields(PathsConfig):
        value = getattr(config.paths, f.name)
        if value is not None:
            setattr(config.paths, f.name, str((base / value).resolve()) if not Path(value).is_absolute() else value)
    return config
