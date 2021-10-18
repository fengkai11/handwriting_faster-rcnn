from torch import nn
import torch
from torch.nn import functional as F
from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator
import numpy as np
class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels = 512,
                 mid_channels = 512,
                 ratios = [0.5,1,2],
                 anchor_scales = [8,16,32],
                 feat_stride = 16,
                 proposal_creator_params = dict()):
        super(RegionProposalNetwork,self).__init__()
        self.anchor_base = generate_anchor_base(ratios = ratios,anchor_scales= anchor_scales)
        self.feat_stride = feat_stride
        self.proposal_layer =ProposalCreator(self,**proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.score = nn.Conv2d(mid_channels,n_anchor*2,1,1,0)#2 class
        self.loc = nn.Conv2d(mid_channels,n_anchor*4,1,1,0)
        self.conv1 = nn.Conv2d(in_channels,mid_channels,3,1,1)


    #build an anchor map
    def enumerate_shifted_anchor(self,anchor_base,feat_stride,height,width):
        y = np.arange(0,height*feat_stride,feat_stride)
        x = np.arange(0,width*feat_stride,feat_stride)
        mx,my = np.meshgrid(x,y)
        shift = np.stack((my.ravel(),mx.ravel(),my.ravel(),mx.ravel()),axis=1)
        anchor = anchor_base.reshape(1,-1,4)+shift.reshape(1,-1,4).transpose((1,0,2))
        anchor = anchor.reshape((-1,4))#anchor_base...
    def forward(self, x, img_size, scale =1, n_anchor=None):
        n,_,hh,ww = x.shape
        anchor = self.enumerate_shifted_anchor(self.anchor_base,self.feat_stride,hh,ww)#call function
        n_anchor = anchor.shape[0]//(hh*ww)
        h = F.relu(self.conv1(x))
        rpn_loc = self.loc(x)
        rpn_score = self.score(x)
        rpn_loc = rpn_loc.permute(0,2,3,1).contiguous().view(n,-1,4)
        rpn_score = rpn_score.permute(0,2,3,1).contiguous()
        rpn_softmax_score = F.softmax(rpn_score.view(n,hh,ww,n_anchor,2),dim =4)
        rpn_fg_scores = rpn_softmax_score[:,:,:,:,1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n,-1)
        rpn_softmax_score = rpn_softmax_score.view(n,-1,2)
        rois = []
        rois_indices = []
        for i in range(n):
            roi = self.proposal_layer(rpn_loc[i].cpu().numpy(),
                                      rpn_fg_scores.cpu().numpy(),
                                      anchor,
                                      img_size,
                                      scale = scale)
            batch_index = i*np.ones(len(roi))
            rois.append(roi)
            rois_indices.append(batch_index)
        return rpn_loc,rpn_score,rois,rois_indices,anchor

if __name__ == '__main__':
    model = RegionProposalNetwork()
    from PIL import Image
    img = Image.open(r'D:\tmp.jpg')
    img = img.resize((512, 512))
    im_arr = np.array(img)
    im_arr = im_arr.transpose(2, 1, 0)
    im_t = torch.from_numpy(im_arr)
    im_t = im_t.unsqueeze(0)
    im_t = im_t.float()  # input must be float tensor
    model.forward(im_t,(512,512))
    print(model)





