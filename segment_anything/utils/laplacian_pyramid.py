import cv2
import numpy as np
import torch.nn as nn


class LaplacianPyramid(nn.Module):
    def __init__(self, level: int = 4):
        super().__init__()
        self.level = level

    def build_laplacian_pyramid(self, image):
        self.gaussian_pyramid = [image]
        for i in range(self.levels - 1):
            image = cv2.pyrDown(image)
            self.gaussian_pyramid.append(image)
        
        laplacian_pyramid = [self.gaussian_pyramid[self.levels - 1]]
        for i in range(self.levels - 1, 0, -1):
            expanded = cv2.pyrUp(self.gaussian_pyramid[i])
            laplacian = cv2.subtract(self.gaussian_pyramid[i - 1], expanded)
            laplacian_pyramid.append(laplacian)
        return self.gaussian_pyramid
    
    def blend_images(self, image1, image2):
        laplacian_pyramid1 = self.build_laplacian_pyramid(image1, self.levels)
        laplacian_pyramid2 = self.build_laplacian_pyramid(image2, self.levels)
        
        blended_pyramid = []
        for lap1, lap2 in zip(laplacian_pyramid1, laplacian_pyramid2):
            blended = np.where(np.abs(lap1) > np.abs(lap2), lap1, lap2)
            blended_pyramid.append(blended)
        
        blended_image = blended_pyramid[0]
        for i in range(1, self.levels - 1):
            blended_image = cv2.pyrUp(blended_image)
            blended_image = cv2.add(blended_pyramid[i], blended_image) // 2

        return blended_image
    
    def forward(self, low_level, high_level):
        pass