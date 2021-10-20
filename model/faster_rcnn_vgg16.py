from __future__ import absolute_import#import package from system site package
from utils.config import opt
from torchvision.ops import RoIPool
from torchvision.models import vgg16
from torch import nn
import torch
from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork
import torch as t
#from torchvision load model;
def decom_vgg16():
    if opt.caffe_pretrain:
            model = vgg16(pretrained=True)
            # if opt.load_path:
            #     model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)
    #TODO: save model
    #decompose vgg
    features = list(model.features)[:30]
    classifier = list(model.classifier)
    del classifier[6]
    classifier = nn.Sequential(*classifier)
    #freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requirs_grad = False
    return nn.Sequential(*features),classifier
class FasterRCNNVGG16(FasterRCNN):#how to init FasterRCNN
    def __init__(self,n_fg_class = 20,ratios = [0.5,1,2],anchor_scales = [8,16,32],feat_stride = 16):
        extractor,classifier = decom_vgg16()
        rpn = RegionProposalNetwork(512,
                                    512,
                                    ratios=ratios,
                                    anchor_scales = anchor_scales,
                                    feat_stride = feat_stride)
        head = VGG16ROIHead(n_fg_class+1,
                            roi_size= 7,
                            spatial_scale=(1/self.feat_stride),
                            classifier = classifier)

class VGG16ROIHead(nn.Module):
    def __init__(self,n_class,roi_size,spatial_scale,classifier):
        self.item_loc = nn.Linear(4096,n_class*4)
        self.item_cls = nn.Linear(4096,n_class*2)
        self.norm_init(self.item_loc,0,0.001)
        self.norm_init(self.item_cls,0,0.01)
        self.roi_size = roi_size
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.RoIPool = RoIPool((self.roi_size,self.roi_size),self.spatial_scale)
        self.classifier = classifier
    def norm_init(self,m,mean,stddev,truncated = False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean,stddev)
            m.bias.data.zero_()

    def forward(self,x,roi,roi_indice):
        roi = torch.from_numpy(roi)
        roi_indice = torch.from_numpy(roi_indice)
        rois = torch.cat([roi_indice[:,None],roi],dim = 1)
        rois = rois[:,[0,2,1,4,3]]
        rois = rois.contiguous()
        pool = self.RoIPool(x,rois)
        pool = pool.view(rois.size()[0],-1)
        fc = self.classifier(pool)
        roi_loc = self.item_loc(fc)
        roi_cls = self.item_cls(fc)
        return roi_loc,roi_cls
if __name__ == '__main__':
    model = decom_vgg16()
    print(model)



