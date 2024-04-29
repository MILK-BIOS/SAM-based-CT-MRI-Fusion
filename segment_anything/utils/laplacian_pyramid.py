import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


class LaplacianPyramid(nn.Module):
    def __init__(self, levels: int = 4, device: str='cuda'):
        super().__init__()
        self.levels = levels
        self.device = device

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
    
    def blend_images(self, image1, image2):
        laplacian_pyramid1 = self.laplacian_pyramid_CT
        laplacian_pyramid2 = self.laplacian_pyramid_MRI
        
        blended_pyramid = []
        for lap1, lap2 in zip(laplacian_pyramid1, laplacian_pyramid2):
            blended = np.where(np.abs(lap1) > np.abs(lap2), lap1, lap2)
            blended_pyramid.append(blended)
        
        blended_image = blended_pyramid[0]
        for i in range(1, self.levels - 1):
            blended_image = cv2.pyrUp(blended_image)
            blended_image = cv2.add(blended_pyramid[i], blended_image) // 2

        return blended_image
    
    def forward(self, img):
        max_values, max_indices = torch.max(torch.stack([self.laplacian_pyramid_CT, self.laplacian_pyramid_MRI]), dim=0)
        max_values = torch.sum(max_values, dim=0)
        assert max_values.shape == img.shape, f"Max value has a shape of {max_values.shape}, but got img shape of {img.shape}"
        blended_image = torch.div((max_values + img), 5)
        return blended_image