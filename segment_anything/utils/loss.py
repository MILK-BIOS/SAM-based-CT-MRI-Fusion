import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from pytorch_msssim import ms_ssim, MS_SSIM


class ContrasiveStructureLoss(nn.Module):
    def __init__(self, 
                 data_range=1, 
                 win_size=11, 
                 win_sigma=1.5, 
                 k1=0.01, 
                 k2=0.03, 
                 eps=1e-8, 
                 reduction='mean',
                 use_contrasive = False,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.classification_loss = nn.CrossEntropyLoss()
        self.illumination_loss = IlluminationLoss()
        self.psnr_loss = PSNRLoss()
        self.spa = L_spa()
        self.use_contrasive = use_contrasive

    def forward(self, 
                CT_pred: Tensor, 
                MRI_pred:Tensor,
                merged: Tensor, 
                encoded_CT: Tensor, 
                encoded_MRI: Tensor, 
                CT_target: Tensor,
                MRI_target: Tensor,
                origin_CT: Tensor, 
                origin_MRI: Tensor,):
        # SSIM
        ssim_loss_CT = 1 - ms_ssim(origin_CT, merged, data_range=1, size_average=True) + 0.01 * max(50 - self.psnr_loss(origin_CT, merged), 0)
        ssim_loss_MRI = 1 - ms_ssim(origin_MRI, merged, data_range=1, size_average=True) + 0.01 * max(50 - self.psnr_loss(origin_MRI, merged), 0)
        spa = self.spa(origin_CT, merged) + self.spa(origin_MRI, merged)
        # Cross Entropy Loss
        classification_loss = self.classification_loss(CT_pred, CT_target) + self.classification_loss(MRI_pred, MRI_target)

        # Illumination Loss
        illuminationLoss = self.illumination_loss(merged)
        loss = ssim_loss_CT + 5 * ssim_loss_MRI + classification_loss + 0.1 * max(8 - illuminationLoss, 0) + spa
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
        loss = is_same_class * dis +  2 * (1 - is_same_class) * torch.max(self.threshold - dis, torch.zeros(len(dis)).to(device)) + torch.max(25-dis_CT, torch.zeros(len(dis)).to(device)) + torch.max(25-dis_MRI, torch.zeros(len(dis)).to(device))
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
    
class L_spa(nn.Module):
 
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org , enhance):
        b,c,h,w = org.shape
 
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
 
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
 
        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)
 
 
        D_org_letf = torch.nn.functional.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = torch.nn.functional.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = torch.nn.functional.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = torch.nn.functional.conv2d(org_pool , self.weight_down, padding=1)
 
        D_enhance_letf = torch.nn.functional.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = torch.nn.functional.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = torch.nn.functional.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = torch.nn.functional.conv2d(enhance_pool , self.weight_down, padding=1)
 
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
 
        return E.mean()