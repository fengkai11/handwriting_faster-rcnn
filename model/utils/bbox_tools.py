from __future__ import division
import numpy as np

# w = base/sqrt(ratio)
# h = base*sqrt(ratio)
# h/w = ratio
def generate_anchor_base(base_size = 16, ratios = [0.5,1,2], anchor_scales = [8,16,32]):
    px = base_size/2
    py = base_size/2
    anchor_base = np.zeros((len(ratios)*len(anchor_scales),4))
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            index = i*len(anchor_scales)+j
            w = base_size*anchor_scales[j]/np.sqrt(ratios[i])
            h = base_size * anchor_scales[j]* np.sqrt(ratios[i])
            anchor_base[index, 0] = px-h/2
            anchor_base[index, 1] = py-w/2
            anchor_base[index, 2] = px+h/2
            anchor_base[index, 3] = py+w/2
    return anchor_base
#decode bbox from loc and anchor
# cx = lcx*base+anchorx
# w = exp(locw)*anchorw
# TODO:rewrite it by tensor
def loc2bbox(anchor,loc):
    anchor_height = anchor[:,2]-anchor[:,0]
    anchor_width = anchor[:,3]-anchor[:,1]
    anchor_ctr_y = (anchor[:,0]+anchor[:,2])/2
    anchor_ctr_x = (anchor[:,1]+anchor[:,3])/2
    dy = loc[:,0::4]
    dx = loc[:,1::4]
    dh = loc[:,2::4]
    dw = loc[:,3::4]
    ctr_y = dy*anchor_height[:,np.newaxis]+anchor_ctr_y[:,np.newaxis]
    ctr_x = dx*anchor_width[:,np.newaxis]+anchor_ctr_x[:,np.newaxis]
    h = np.exp(dh)*anchor_height[:,np.newaxis]
    w = np.exp(dw)*anchor_width[:,np.newaxis]
    dst_bbox = np.zeros_like(loc)
    dst_bbox[:,0::4] = ctr_y-0.5*h
    dst_bbox[:,1::4] = ctr_x-0.5*w
    dst_bbox[:,2::4] = ctr_y+0.5*h
    dst_bbox[:,3::4] = ctr_x+0.5*w
    return dst_bbox
if __name__ == "__main__":
    tmp = generate_anchor_base()
