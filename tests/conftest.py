"""Synthetic flight, top view and support bank; an injected backend replaces FSS-SAM3 (no GPU, no weights)."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from triffid_uav.models.base import ClassPrediction  # noqa: E402
from triffid_uav.ontology import CLASSES  # noqa: E402
from triffid_uav.supports import BANK_SCHEMA  # noqa: E402


class FakeBackend:
    """Deterministic 5x6 square per class; records preparation and calls."""
    def __init__(self):
        self.calls, self.prepared, self.shots = 0, 0, None

    def prepare(self, classes, supports):
        supports.require_classes(classes)
        self.prepared += 1
        self.shots = supports.shots
        self.classes = tuple(classes)

    def predict(self, rgb, classes):
        self.calls += 1
        mask = np.zeros(rgb.shape[:2], bool)
        mask[4:9, 7:13] = True
        return [ClassPrediction(c.id, mask.copy(), np.where(mask, .8, 0).astype(np.float32)) for c in classes]

    def metadata(self):
        return {'name': 'fake', 'shots': self.shots}


@pytest.fixture
def fake_backend():
    return FakeBackend()


def srt_block(index, seconds, lat, lon, yaw, pitch):
    return (f'{index}\n00:00:{seconds:02},000 --> 00:00:{seconds:02},100\n'
            f'<font size="28">FrameCnt: {index}, DiffTime: 33ms\n'
            f'[latitude: {lat}] [longitude: {lon}] [rel_alt: 50.0 abs_alt: 150.0] '
            f'[gb_yaw: {yaw} gb_pitch: {pitch} gb_roll: 0.0] </font>')


@pytest.fixture
def project(tmp_path):
    """A project root laid out like the repo: inputs/, input_topview/, supports/, outputs/."""
    root = tmp_path / 'project'
    inputs, topview, supports = root / 'inputs', root / 'input_topview', root / 'supports'
    for folder in (inputs, topview, supports):
        folder.mkdir(parents=True)
    # 10 fps, 80 frames; frame i has brightness i so tests can tell frames apart.
    writer = cv2.VideoWriter(str(inputs / 'flight.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    for i in range(80):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    # One SRT block per second around a circle; FrameCnt = 10 * second + 1.
    blocks = []
    for second, degrees in enumerate(range(0, 360, 45)):
        a = math.radians(degrees)
        blocks.append(srt_block(second * 10 + 1, second, 38 + .001 * math.cos(a),
                                23 + .001 * math.sin(a) / math.cos(math.radians(38)), (degrees + 180) % 360, -45))
    (inputs / 'flight.srt').write_text('\n\n'.join(blocks))
    # Mini topview SRT naming FrameCnt 31 (video frame 30).
    (topview / 'topview.srt').write_text(srt_block(31, 3, 38.0005, 23.0005, 0, -90))
    # Support bank: five synthetic image/mask pairs for every class.
    entries = {}
    for item in CLASSES:
        entries[item.name] = []
        for slot in range(1, 6):
            image, mask = supports / f'{item.id}_{slot}_image.png', supports / f'{item.id}_{slot}_mask.png'
            cv2.imwrite(str(image), np.full((20, 30, 3), slot * 20 + item.id, np.uint8))
            foreground = np.zeros((20, 30), np.uint8)
            foreground[4:16, 5:25] = 255
            cv2.imwrite(str(mask), foreground)
            entries[item.name].append({'id': f'example_{slot:02d}', 'image': image.name, 'mask': mask.name})
    (supports / 'bank.json').write_text(json.dumps({'schema_version': BANK_SCHEMA, 'classes': entries}))
    return root
