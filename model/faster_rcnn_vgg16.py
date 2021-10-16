from __future__ import absolute_import#import package from system site package
from utils.config import opt
from torchvision.models import vgg16
from torch import nn
import torch as t
#from torchvision load model;
def decom_vgg16():
    if opt.caffe_pretrain:
            model = vgg16(pretrained=False)
            if opt.load_path:
                model.load_state_dict(t.load(opt.caffe_pretrain_path))
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
if __name__ == '__main__':
    model = decom_vgg16()



