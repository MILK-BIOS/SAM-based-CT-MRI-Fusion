import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from ..modeling.transformer import Attention
from ..modeling.common import LayerNorm2d


class LaplacianPyramid(nn.Module):
    def __init__(self, levels: int = 4, 
                 device: str='cuda', 
                 in_chans: int = 1, 
                 embed_dim: int = 768,
                 patch_size: int = 16,
                ):
        super().__init__()
        self.levels = levels
        self.device = device
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0).to(device)
        self.attention_mode = Attention(embed_dim, 16).to(device)
        self.attention_fusion = Attention(embed_dim, 16).to(device)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=4, stride=4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, in_chans, 1)
        ).to(device)

    def build_laplacian_pyramid_CT(self, image: torch.Tensor, for_train: bool = True):
        self.gaussian_pyramid_CT = [image]
        for i in range(self.levels - 1):
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.gaussian_pyramid_CT.append(image)
        
        self.laplacian_pyramid_CT = [self.gaussian_pyramid_CT[self.levels - 1]]
        for i in range(self.levels - 1, 0, -1):
            expanded = F.interpolate(self.gaussian_pyramid_CT[i], scale_factor=2.0, mode='bilinear', align_corners=False)
            laplacian = self.gaussian_pyramid_CT[i - 1] - expanded
            self.laplacian_pyramid_CT.append(laplacian)

        if for_train:
            target_size = self.laplacian_pyramid_CT[-1].shape[-2:]
            resized_pyramid = [F.interpolate(img, size=target_size, mode='bilinear', align_corners=False) for img in self.laplacian_pyramid_CT]
            self.laplacian_pyramid_CT = torch.stack(resized_pyramid, dim=0)
            self.laplacian_pyramid_CT = self.laplacian_pyramid_CT.to(self.device)

        return self.laplacian_pyramid_CT
    
    def build_laplacian_pyramid_MRI(self, image: torch.Tensor, for_train: bool = True):
        self.gaussian_pyramid_MRI = [image]
        for i in range(self.levels - 1):
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.gaussian_pyramid_MRI.append(image)
        
        self.laplacian_pyramid_MRI = [self.gaussian_pyramid_MRI[self.levels - 1]]
        for i in range(self.levels - 1, 0, -1):
            expanded = F.interpolate(self.gaussian_pyramid_MRI[i], scale_factor=2.0, mode='bilinear', align_corners=False)
            laplacian = self.gaussian_pyramid_MRI[i - 1] - expanded
            self.laplacian_pyramid_MRI.append(laplacian)

        if for_train:
            target_size = self.laplacian_pyramid_MRI[-1].shape[-2:]
            resized_pyramid = [F.interpolate(img, size=target_size, mode='bilinear', align_corners=False) for img in self.laplacian_pyramid_MRI]
            self.laplacian_pyramid_MRI = torch.stack(resized_pyramid, dim=0)
            self.laplacian_pyramid_MRI = self.laplacian_pyramid_MRI.to(self.device)

        return self.laplacian_pyramid_MRI
    
    def blend_images(self, img, pyr_list):
        return 
    
    def forward(self, img):
        L, B, C, H, W = self.laplacian_pyramid_CT.shape
        img = torch.div(img, 5)
        embedded_CT = self.patch_embedding(self.laplacian_pyramid_CT.view(-1, C, H, W))
        embedded_MRI = self.patch_embedding(self.laplacian_pyramid_MRI.view(-1, C, H, W))
        embedded_CT = embedded_CT.permute(0,2,3,1).reshape(-1, H//self.patch_size * W//self.patch_size, self.embed_dim)
        embedded_MRI = embedded_MRI.permute(0,2,3,1).reshape(-1, H//self.patch_size * W//self.patch_size, self.embed_dim)

        fusion_pyramid = self.attention_mode(q=embedded_MRI, k=embedded_CT, v=embedded_MRI).view(-1, H//self.patch_size, W//self.patch_size, self.embed_dim).permute(0, 3, 1, 2)
        upscaled = self.output_upscaling(fusion_pyramid).reshape(L, B, H, W, -1).permute(0, 1, 4, 2, 3)
        max_values, max_indices = torch.max(torch.stack([self.laplacian_pyramid_CT, self.laplacian_pyramid_MRI]), dim=0)
        upscaled = upscaled + max_values
        fusion_img = torch.sum(upscaled, dim=0)
        assert fusion_img.shape == img.shape, f"Max value has a shape of {fusion_img.shape}, but got img shape of {img.shape}"
        blended_image = torch.div((fusion_img + img), 1)
        return blended_image
    

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int =768):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, CT, MRI):
        k = self.Wk(MRI)
        v = self.Wv(MRI)
        q = self.Wq(CT)