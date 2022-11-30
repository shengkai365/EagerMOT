from __future__ import annotations
from typing import Optional
from inputs import bbox

class Detection2D(bbox.ProjectsToCam):
    def __init__(self,
                 bbox: bbox.Bbox2d,
                 cam: str,
                 score: float,
                 seg_class_id: int,
                 *,
                 mask=None,
                 mask_decoded=None,
                 reid=None):
        self.bbox = bbox
        self.cam = cam
        self.score = score
        self.seg_class_id = seg_class_id
        self.mask = mask
        self.mask_decoded = mask_decoded
        self.reid = reid

    def bbox_2d_in_cam(self, cam: str) -> Optional[bbox.Bbox2d]:
        return self.bbox if cam == self.cam else None
