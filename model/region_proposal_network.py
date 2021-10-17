from torch import nn

class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels = 512,
                 mid_channels = 512,
                 ratios = [0.5,1,2],
                 anchor_scales = [8,16,32],
                 feat_stride = 16,
                 proposal_creator_params = dict()):
        super(RegionProposalNetwork,self).__init__()

        i = 1
    def forward(self,x,img_size,scale =1):
        i =1
