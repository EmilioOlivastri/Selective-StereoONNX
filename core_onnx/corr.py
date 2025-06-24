import torch
import torch.nn.functional as F

import corr_sampler

class CorrBlockFast1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # all pairs correlation
        corr = corr = self.corr(fmap1, fmap2)
        B, H, W1, _, W2 = corr.shape
        corr = corr.view(B * H * W1, 1, 1, W2)  # [B*H*W1, 1, 1, W2]

        for i in range(self.num_levels):
            downsampled = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            downsampled = downsampled.view(B, H, W1, -1, downsampled.shape[-1])
            self.corr_pyramid.append(downsampled)
            corr = downsampled.view(B * H * W1, 1, 1, downsampled.shape[-1])

    def __call__(self, coords):
        B, _, H, W = coords.shape
        coords_x = coords[:, [0]]  # [B, 1, H, W]
        
        out_pyramid = []
        for i in range(self.num_levels):
            corr_lvl = self.corr_pyramid[i]  # [B, H, W1, dim, W2_i]
            corr_lvl = corr_lvl.squeeze(3)   # [B, H, W1, W2_i]
            W2 = corr_lvl.shape[-1]

            r = self.radius
            dx = torch.arange(-r, r + 1, device=coords.device).view(1, 2 * r + 1, 1, 1)  # [1, 2r+1, 1, 1]
            coords_x_lvl = coords_x / (2 ** i)
            sample_coords = coords_x_lvl + dx
            sample_coords = sample_coords.clamp(0, W2 - 1)
            sample_coords = sample_coords.long()

            corr_lvl = corr_lvl.permute(0, 2, 1, 3)  # [B, W1, H, W2]
            gathered = []
            for j in range(2 * r + 1):
                idx = sample_coords[:, j, :, :]  # [B, H, W]
                idx = idx.unsqueeze(1).expand(-1, corr_lvl.shape[1], -1, -1)  # [B, W1, H, W]
                gathered_j = torch.gather(corr_lvl, dim=-1, index=idx)
                gathered.append(gathered_j)

            corr_slice = torch.stack(gathered, dim=1)  # [B, 2r+1, W1, H, W]
            corr_slice = corr_slice.permute(0, 1, 3, 4, 2)
            corr_slice = corr_slice.mean(dim=-1)  # [B, 2r+1, H, W]\
            out_pyramid.append(corr_slice)
            
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape

        # Reshape for batch matrix multiplication
        fmap1 = fmap1.permute(0, 2, 3, 1).reshape(B * H, W1, D)  # (B*H, W1, D)
        fmap2 = fmap2.permute(0, 2, 3, 1).reshape(B * H, W2, D)  # (B*H, W2, D)

        # Transpose fmap2 for matmul
        corr = torch.bmm(fmap1, fmap2.transpose(1, 2))  # (B*H, W1, W2)

        # Reshape to (B, H, W1, 1, W2)
        corr = corr.view(B, H, W1, 1, W2).contiguous()

        scale = torch.sqrt(fmap1.new_ones(1) * D).float() # <-- Extra 2 lines added by me
        return corr / scale # <-- Extra 2 lines added by me


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)
        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, 1, w2)
        self.corr_pyramid.append(corr)

        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 1]
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # [B*H*W, 1, 1, W']
            W_corr = corr.shape[-1]

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device, dtype=coords.dtype)  # [2r+1]
            dx = dx.view(1, -1, 1, 1)  # [1, 2r+1, 1, 1]

            # coords scaled to current level resolution
            coords_lvl = coords.view(batch * h1 * w1, 1, 1, 1) / (2 ** i)  # [B*H*W, 1, 1, 1]
            x = coords_lvl + dx  # [B*H*W, 2r+1, 1, 1]
            y = torch.zeros_like(x)

            grid = torch.cat([x, y], dim=-1)  # [B*H*W, 2r+1, 1, 2]
            
            norm = W_corr - 1
            grid = grid / norm * 2 - 1  # normalize to [-1, 1]

            sampled = F.grid_sample(corr, grid, mode='bilinear', align_corners=True)  # [B*H*W, 1, 1, 2r+1]
            sampled = sampled.view(batch, h1, w1, -1)
            out_pyramid.append(sampled)

        out = torch.cat(out_pyramid, dim=-1)  # [B, H, W, C]
        return out.permute(0, 3, 1, 2).contiguous().float()  # [B, C, H, W]

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)  # [B, H, W1, W2]
        corr = corr.view(B, H, W1, 1, W2)
        scale = torch.sqrt(torch.tensor(D, device=fmap1.device, dtype=fmap1.dtype))
        return corr / scale