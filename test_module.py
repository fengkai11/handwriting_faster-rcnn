from model.faster_rcnn_vgg16 import decom_vgg16
from model.region_proposal_network import RegionProposalNetwork
from torch import nn
import torch
from PIL import Image
import numpy as np
#1. test extractor
#FIXME:class inherit
class RPN(nn.Module):
    def __init__(self,extractor,RegionProposalNetwork):
        super(RPN,self).__init__()#@1
        self.extractor = extractor
        self.head = RegionProposalNetwork
    def forward(self,x):
        img_size = x.size()[2:]
        x = self.extractor(x)
        result = self.head(x,img_size,1)
        return result
class Test(RPN):
    def __init__(self):
        features = decom_vgg16()
        rpn = RegionProposalNetwork()
        super(Test,self).__init__(features,rpn)
#2. test region proposal
def load_img(path):
    img = Image.open(path)
    img = img.resize((512, 512))
    im_arr = np.array(img)
    im_arr = im_arr.transpose(2, 1, 0)
    im_t = torch.from_numpy(im_arr)
    im_t = im_t.unsqueeze(0)
    im_t = im_t.float()  # input must be float tensor
    return im_t
#cannot assign module before Module.__init__() call
# @1. inhert nn.module should call init_first
if __name__ == '__main__':
    print("hello world")
    '''
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
    '''
    # 2. test region proposal
    model = Test()
    model = model.state_dict(torch.load(r'D:\BaiduNetdiskDownload\simple-faster-rcnn\fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth',map_location = torch.device('cpu')))
    im_t = load_img(r'D:/tmp.jpg')
    model.forward(im_t)