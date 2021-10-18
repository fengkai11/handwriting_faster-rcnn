from __future__ import absolute_import#import package from system site package
from utils.config import opt
from torchvision.models import vgg16
from torch import nn
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
    #freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requirs_grad = False
    return nn.Sequential(*features)
class FasterRCNNVGG16(FasterRCNN):#how to init FasterRCNN
    def __init__(self,n_fg_class = 20,ratios = [0.5,1,2],anchor_scales = [8,16,32],feat_stride = 16):
        extractor = decom_vgg16()
        rpn = RegionProposalNetwork(512,
                                    512,
                                    ratios=ratios,
                                    anchor_scales = anchor_scales,
                                    feat_stride = feat_stride)


if __name__ == '__main__':
    model = decom_vgg16()
    print(model)



