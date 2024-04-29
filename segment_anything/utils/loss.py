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
                 reduction='mean',
                 device='cuda'):
        super().__init__()
        self.device = device
        self.ssim_loss = SSIM(data_range=data_range, 
                             k1=k1, 
                             k2=k2).to(device)
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrasive_loss = ContrasiveLoss()
        self.illumination_loss = IlluminationLoss()
        self.psnr_loss = PSNRLoss()

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
        ssim_loss_CT = 1 - self.ssim_loss(target=origin_CT, preds=merged) + 0.02 * max(50 - self.psnr_loss(origin_CT, merged), 0)
        ssim_loss_MRI = 1 - self.ssim_loss(target=origin_MRI, preds=merged) + 0.02 * max(50 - self.psnr_loss(origin_MRI, merged), 0)

        # Cross Entropy Loss
        classification_loss = self.classification_loss(CT_pred, CT_target) + self.classification_loss(MRI_pred, MRI_target)

        # Contrasive Loss
        is_same_class = (CT_target == MRI_target).float()
        contrasive_loss = self.contrasive_loss(encoded_CT, encoded_MRI, is_same_class, self.device)

        # Illumination Loss
        illuminationLoss = self.illumination_loss(merged)

        loss = ssim_loss_CT + 5 * ssim_loss_MRI + classification_loss + 0.1 * contrasive_loss + 0.1 * max(3 - illuminationLoss, 0) + 0.01 * max(150 - torch.mean(merged), 0)
        return loss
    
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, origin_CT, origin_MRI, merged):
        pass


class PSNRLoss(nn.Module):
    def __init__(self, max_pixel=1.0):
        super().__init__()
        self.max_pixel = max_pixel

    def forward(self, origin, merged):
        mse = torch.mean((origin - merged) ** 2)
        if mse == 0:
            return float('inf')

        psnr_value = 20 * torch.log10(self.max_pixel / torch.sqrt(mse))
        return psnr_value.item()  # Return PSNR value as a Python float


class ContrasiveLoss(nn.Module):
    def __init__(self, threshold: int=100):
        super().__init__()
        self.threshold = threshold

    def forward(self, encoded_CT, encoded_MRI, is_same_class, device='cuda'):
        is_same_class = is_same_class.int()
        encoded_CT = encoded_CT.view(encoded_CT.shape[0], -1)
        encoded_MRI = encoded_MRI.view(encoded_MRI.shape[0], -1)
        dis_CT = torch.norm(encoded_CT, dim=1, p=2) ** 2
        dis_MRI = torch.norm(encoded_MRI, dim=1, p=2) ** 2
        dis = torch.norm(encoded_CT-encoded_MRI, dim=1, p=2) ** 2
        # Adjust contrastive loss according to the ratio of positive to negetive
        loss = is_same_class * dis +  2 * (1 - is_same_class) * torch.max(self.threshold - dis, torch.zeros(len(dis)).to(device))# + torch.max(25-dis_CT, torch.zeros(len(dis)).to(device)) + torch.max(25-dis_MRI, torch.zeros(len(dis)).to(device))
        loss = torch.mean(loss)
        return loss
    

class IlluminationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        assert len(x.shape) == 4, f'Expect x to be a 4d tensor but got {len(x.shape)}d'
        B, C, H, W = x.shape
        count_h = (H - 1) * W
        count_w = H * (W - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:H-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:W-1]),2).sum()
        loss = (h_tv/count_h + w_tv/count_w) / B
        return loss