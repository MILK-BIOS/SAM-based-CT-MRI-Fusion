import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM


class ContrasiveStructureLoss(nn.Module):
    def __init__(self, 
                 data_range=1.0, 
                 win_size=11, 
                 win_sigma=1.5, 
                 k1=0.01, 
                 k2=0.03, 
                 eps=1e-8, 
                 reduction='mean'):
        super().__init__()
        self.ssim_loss = SSIM(data_range=data_range, 
                             win_size=win_size, 
                             win_sigma=win_sigma, 
                             k1=k1, 
                             k2=k2, 
                             eps=eps, 
                             reduction=reduction)
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrasive_loss = ContrasiveLoss()
        self.illumination_loss = IlluminationLoss()

    def forward(self, 
                CT_pred: Tensor, 
                MRI_pred:Tensor,
                merged: Tensor, 
                encoded_CT: Tensor, 
                encoded_MRI: Tensor, 
                CT_target: Tensor,
                MRI_target: Tensor,
                origin_CT: Tensor, 
                origin_MRI: Tensor):
        # SSIM
        ssim_loss_CT = self.ssim_loss(origin_CT, merged)
        ssim_loss_MRI = self.ssim_loss(origin_MRI, merged)

        # Cross Entropy Loss
        classification_loss = self.classification_loss(CT_pred, CT_target) + self.classification_loss(MRI_pred, MRI_target)

        # Contrasive Loss
        is_same_class = (CT_target == MRI_target)
        contrasive_loss = self.contrasive_loss(encoded_CT, encoded_MRI, is_same_class)

        # Illumination Loss
        illuminationLoss = self.illumination_loss(merged)

        loss = ssim_loss_CT + ssim_loss_MRI + classification_loss + contrasive_loss + illuminationLoss
        return loss
    

class ContrasiveLoss(nn.Module):
    def __init__(self, threshold: int=5000):
        super().__init__()
        self.threshold = threshold

    def forward(self, encoded_CT, encoded_MRI, is_same_class):
        dis = torch.dist(encoded_CT, encoded_MRI) ** 2
        loss = is_same_class * dis + (1 - is_same_class) * max(self.threshold - dis, 0)
        return loss
    

class IlluminationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        assert x.shape == 4, f'Expect x to be a 4d tensor but got {len(x.shape)}d'
        B, C, H, W = x.shape
        count_h = (H - 1) * W
        count_w = H * (W - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:H-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:W-1]),2).sum()
        loss = 2*(h_tv/count_h + w_tv/count_w) / B
        return loss