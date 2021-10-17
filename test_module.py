from model.faster_rcnn_vgg16 import decom_vgg16
from torch import nn
import torch
from PIL import Image
import numpy as np
#test extractor
#FIXME:class inherit
class Extractor(nn.Module):
    def __init__(self,extractor):
        super(Extractor,self).__init__()#@1
        self.extractor = extractor
    def forward(self,x):
        x = self.extractor(x)
        return x
class Test(Extractor):
    def __init__(self):
        features = decom_vgg16()
        super(Test,self).__init__(features)

#cannot assign module before Module.__init__() call
# @1. inhert nn.module should call init_first
if __name__ == '__main__':
    print("hello world")
    model = Test()#function build or init a class
    # print parameter shape;
    layer_name = list(model.state_dict().keys())
    for i in range(0,len(layer_name),2):
        print(layer_name[i],
              model.state_dict()[layer_name[i]].size(),
              model.state_dict()[layer_name[i+1]].size())
    img = Image.open(r'D:\tmp.jpg')
    img = img.resize((512,512))
    im_arr = np.array(img)
    im_arr = im_arr.transpose(2,1,0)
    im_t = torch.from_numpy(im_arr)
    im_t = im_t.unsqueeze(0)
    im_t = im_t.float()#input must be float tensor
    print(im_t.size())
    out = model.forward(im_t)#only the return value can be check
    # t = 0
#argument 'input' (position 1) must be Tensor, not numpy.ndarray


# print feature map