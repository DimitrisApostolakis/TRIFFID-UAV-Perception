"""FSS-SAM3 few-shot segmentation backend (https://github.com/WongKinYiu/FSS-SAM3).

Layouts follow the official model/evaluate.py:
  1-shot: query on top, support below (support_ratio_1shot of the canvas height).
  5-shot: "l_shape" - three supports across the top, two down the left, query bottom-right.
Each support contributes its tight mask bounding box as a positive visual prompt.
Shots are whatever the support selection holds (1 or 5); nothing here is fixed to one.
"""
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from .base import ClassPrediction


def one_shot_canvas(support_rgb,support_mask,query_rgb,size=1008,ratio=.6):
    ys,xs=np.nonzero(support_mask)
    if not len(xs):raise ValueError('Empty visual support mask')
    support_h=int(size*ratio);query_h=size-support_h
    canvas=Image.new('RGB',(size,size))
    canvas.paste(Image.fromarray(query_rgb).resize((size,query_h),Image.Resampling.BILINEAR),(0,0))
    canvas.paste(Image.fromarray(support_rgb).resize((size,support_h),Image.Resampling.BILINEAR),(0,query_h))
    height,width=support_mask.shape
    x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
    box=[(x0+x1)/2/width,(query_h+(y0+y1)/2/height*support_h)/size,
         (x1-x0)/width,(y1-y0)/height*support_h/size]
    return canvas,box,(0,0,size,query_h)


def five_shot_canvas(references,query_rgb,size=1008,ratio=.3):
    """Official evaluate.py l_shape: three top supports, two left, query bottom-right."""
    if len(references)!=5:raise ValueError('Five-shot canvas requires exactly five supports')
    thick=int(round(size*ratio));target=size-thick;top_w=size//3;left_h=target//2
    layouts=[(0,0,top_w,thick),(top_w,0,top_w,thick),
             (top_w*2,0,size-top_w*2,thick),(0,thick,thick,left_h),
             (0,thick+left_h,thick,target-left_h)]
    canvas=Image.new('RGB',(size,size));boxes=[]
    for (rgb,mask),(x,y,w,h) in zip(references,layouts):
        ys,xs=np.nonzero(mask)
        if not len(xs):raise ValueError('Empty visual support mask')
        canvas.paste(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR),(x,y))
        mh,mw=mask.shape
        x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
        boxes.append([(x+(x0+x1)/2/mw*w)/size,(y+(y0+y1)/2/mh*h)/size,
                      (x1-x0)/mw*w/size,(y1-y0)/mh*h/size])
    canvas.paste(Image.fromarray(query_rgb).resize((target,target),Image.Resampling.BILINEAR),(thick,thick))
    return canvas,boxes,(thick,thick,target,target)


class FssSam3Backend:
    name = 'fss-sam3'

    def __init__(self, repo, checkpoint, bpe_path, *, device='cuda', threshold=.5, prompt='visual',
                 canvas_size=1008, support_ratio_1shot=.6, support_ratio_5shot=.3):
        if prompt not in ('visual', 'text+visual'):
            raise ValueError("model.prompt must be 'visual' or 'text+visual'")
        if not 0 < threshold < 1:
            raise ValueError('model.threshold must be in (0, 1)')
        if canvas_size < 64 or not (0 < support_ratio_1shot < 1 and 0 < support_ratio_5shot < .5):
            raise ValueError('Invalid canvas size or support ratios')
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.bpe_path = Path(bpe_path).resolve()
        self.device, self.threshold, self.prompt = device, threshold, prompt
        self.canvas_size = canvas_size
        self.support_ratio_1shot, self.support_ratio_5shot = support_ratio_1shot, support_ratio_5shot
        self.diagnostics = []
        self.model = None
        self.references = {}
        self.shots = None
        for path in (self.repo / 'sam3/model_builder.py', self.checkpoint, self.bpe_path):
            if not path.is_file():
                raise ValueError(f'Missing FSS-SAM3 asset: {path}')

    @property
    def use_text(self):
        return self.prompt == 'text+visual'

    def prepare(self, classes, supports):
        """Load SAM3 once and cache the support images/masks for every class of the run."""
        import torch
        supports.require_classes(classes)
        if supports.shots not in (1, 5):
            raise ValueError('FSS-SAM3 layouts exist for 1 or 5 supports per class')
        self.shots = supports.shots
        self.torch = torch
        self.references = {c.id: [s.load() for s in supports.for_class(c)] for c in classes}
        if self.model is not None:
            return
        sys.path.insert(0, str(self.repo))
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        if Path(sam3.__file__).resolve().parent != self.repo / 'sam3':
            raise RuntimeError('Another SAM3 installation shadows the selected FSS-SAM3 source')
        self.model = sam3.build_sam3_image_model(bpe_path=str(self.bpe_path), device=self.device,
                                                 checkpoint_path=str(self.checkpoint), load_from_HF=False,
                                                 eval_mode=True, compile=False)
        self.model.eval().requires_grad_(False)
        self.processor = Sam3Processor(self.model, resolution=self.canvas_size, device=self.device,
                                       confidence_threshold=self.threshold)

    def _canvas(self, references, rgb):
        if len(references) == 1:
            support, mask = references[0]
            canvas, box, crop = one_shot_canvas(support, mask, rgb, self.canvas_size, self.support_ratio_1shot)
            return canvas, [box], crop
        return five_shot_canvas(references, rgb, self.canvas_size, self.support_ratio_5shot)

    def predict(self, rgb, classes):
        torch = self.torch
        with torch.inference_mode():
            for item in classes:
                canvas, boxes, (x, y, w, h) = self._canvas(self.references[item.id], rgb)
                state = self.processor.set_image(canvas)
                self.processor.reset_all_prompts(state)
                if self.use_text:
                    state = self.processor.set_text_prompt(item.name, state)
                for box in boxes:
                    state = self.processor.add_geometric_prompt(state=state, box=box, label=True)
                self.diagnostics.append(dict(
                    class_name=item.name, accepted_canvas_masks=len(state['masks']),
                    max_canvas_score=float(state['scores'].max().item()) if len(state['scores']) else None,
                    query_foreground_pixels=int(state['masks'][:, :, y:y + h, x:x + w].sum().item())))
                if len(state['masks']):
                    union = state['masks'][:, 0].any(0)[y:y + h, x:x + w].cpu().numpy().astype(np.uint8)
                    probability = state['masks_logits'][:, 0].max(0).values[y:y + h, x:x + w].float().cpu().numpy()
                else:
                    union = np.zeros((h, w), np.uint8)
                    probability = np.zeros((h, w), np.float32)
                shape = (rgb.shape[1], rgb.shape[0])
                restored = cv2.resize(union, shape, interpolation=cv2.INTER_NEAREST).astype(bool)
                probability = cv2.resize(probability, shape, interpolation=cv2.INTER_NEAREST)
                yield ClassPrediction(item.id, restored, probability)

    def _revision(self):
        try:
            return subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def metadata(self):
        return dict(
            name=self.name, source_repository='https://github.com/WongKinYiu/FSS-SAM3',
            reference_revision=self._revision(), checkpoint=str(self.checkpoint), device=self.device,
            precision='float32', shots=self.shots,
            supports_per_class={str(k): len(v) for k, v in self.references.items()},
            canvas_size=self.canvas_size,
            support_fraction=self.support_ratio_1shot if self.shots == 1 else self.support_ratio_5shot,
            layout='query top, support bottom' if self.shots == 1 else 'l_shape: three top, two left, query bottom-right',
            prompting=('class-name text plus visual positive support box' if self.use_text
                       else 'visual positive support box; no class-name text'),
            object_score_threshold=self.threshold, mask_threshold=.5,
            query_restore='crop query from canvas; nearest-neighbour resize to original RGB dimensions',
            inference='independent one-class support/query collage episodes',
            instance_method='FSS mask union followed by 8-connected components',
            native_canvas_diagnostics=self.diagnostics)
