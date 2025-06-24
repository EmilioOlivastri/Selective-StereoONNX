import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_onnx.update import SpatialAttentionExtractor, ChannelAttentionEnhancement, BasicSelectiveMultiUpdateBlock
from core_onnx.extractor import BasicEncoder, MultiBasicEncoder
from core_onnx.corr import CorrBlock1D
from core_onnx.utils.utils import coords_grid

class RAFTONNX(nn.Module):
    def __init__(self, args):
        super(RAFTONNX, self).__init__()
        self.args = args
        
        self.hidden_dim = hdim = self.args.hidden_dim
        self.context_dim = cdim = self.args.hidden_dim

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)  
        self.cnet = MultiBasicEncoder(norm_fn='batch', downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dim)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.hidden_dim)
        self.corr_block = CorrBlock1D
        

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_disp(self, img):
        """ Disp is represented as difference between two coordinate grids Disp = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        # disparity computed as difference: disp = coords1 - coords0
        return coords0, coords1
    
    def upsample_disp(self, disp, mask):
        """ Upsample disp field [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor * H, factor * W)

    def forward(self, images, iters=8):
        """ Estimate disparity between pair of frames """

        image1, image2 = images[:, :3, :, :], images[:, 3:, :, :]

        # Normalize images to [-1, 1] range
        image1 = (2.0 * image1 / 255.0 - 1.0).contiguous()
        image2 = (2.0 * image2 / 255.0 - 1.0).contiguous()

        # run the feature network
        fmap1 = self.fnet(image1).float()
        fmap2 = self.fnet(image2).float()

        corr_fn = self.corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        # run the context network
        cnet = self.cnet(image1)
        net = [torch.tanh(x[0]) for x in cnet]
        inp = [torch.relu(x[1]) for x in cnet]
        inp = [self.cam(x) * x for x in inp]
        att = [self.sam(x) for x in inp]

        coords0, coords1 = self.initialize_disp(net[0])

        up_mask = None
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            disp = coords1 - coords0
            net, up_mask, delta_disp = self.update_block(net, inp, corr, disp, att)

            # F(t+1) = F(t) + \Delta(t)
            coords1 += delta_disp

        # upsample predictions
        return self.upsample_disp(coords1 - coords0, up_mask)
