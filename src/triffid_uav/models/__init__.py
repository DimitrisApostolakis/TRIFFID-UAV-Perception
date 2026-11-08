"""Segmentation backend registry. Add a backend by registering a factory taking the model config."""
from .base import ClassPrediction, SegmentationBackend


def _fss_sam3(config):
    from .fss_sam3 import FssSam3Backend
    return FssSam3Backend(config.repo, config.checkpoint, config.bpe, device=config.device,
                          threshold=config.threshold, prompt=config.prompt, canvas_size=config.canvas_size,
                          support_ratio_1shot=config.support_ratio_1shot,
                          support_ratio_5shot=config.support_ratio_5shot)


BACKENDS = {'fss-sam3': _fss_sam3}


def build_backend(config):
    try:
        factory = BACKENDS[config.backend]
    except KeyError:
        raise ValueError(f'Unknown model.backend {config.backend!r}; available: {sorted(BACKENDS)}') from None
    return factory(config)


__all__ = ['BACKENDS', 'ClassPrediction', 'SegmentationBackend', 'build_backend']
