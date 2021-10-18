from __future__ import absolute_import
from __future__ import division #division float
from torch import nn
#extrat feature generate proposal and roi for pooling;
#
class FasterRCNN(nn.Module):
    def __init__(self,extrator,rpn,head):
        self.head = head
        self.rpn = rpn
        self.extrator = extrator
        i = 0
    def forward(self,x,scale =1):
        img_size = x.shape[2:]
        h = self.extrator(x)
        rpn_locs, rpn_scores, rois, roi_indices,anchor = self.rpn(h,img_size,scale)
        roi_cls_locs,roi_scores = self.head(h,rois,roi_indices)
        return roi_cls_locs,roi_scores,rois,roi_indices
