import numpy as np
import torch
from torchvision.ops import nms

from model.utils.bbox_tools import loc2bbox
# decode bbox
# filter bbox
class ProposalCreator(object):
    def __init__(self,parent_model,
                 nms_thresh = 0.7,
                 n_train_pre_nms = 12000,
                 n_train_post_nms = 2000,
                 n_test_pre_nms = 6000,
                 n_test_post_nms = 300,
                 min_size = 16):
        self.parent_model =parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    def __call__(self, loc, score, anchor, img_size, scale=1, keep_nms=None):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        roi = loc2bbox(anchor,loc)
        roi[:,0::2] = np.clip(roi[:,0::2],0,img_size[0])
        roi[:,1::2] = np.clip(roi[:,1::2],0,img_size[1])
        # remove predicted boxes with either height or width < thresh
        min_size = self.min_size*scale
        hs = roi[:,2] - roi[:,0]
        ws = roi[:,3] - roi[:,1]
        keep = np.where(hs>=min_size and ws>=min_size)[0]
        roi = roi[keep,:]
        score = score[keep,:]
        order = score.ravel().argsort()[::-1]#big2small
        order = order[:n_pre_nms]
        roi = roi[order,:]
        score = score[order]
        keep_nms = nms(torch.from_numpy(roi).cuda(),
                   torch.from_numpy(score).cuda,
                   self.nms_thresh)
        keep_nms = keep_nms[:n_post_nms]
        roi = roi[keep_nms.cpu().numpy()]
        return roi




