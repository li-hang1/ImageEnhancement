import torch.nn as nn
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure


class Loss(nn.Module):
    def __init__(self, lambda_ssim, lambda_per, device, lpips_net='alex'):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lpips = lpips.LPIPS(net=lpips_net).to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.lambda_ssim = lambda_ssim
        self.lambda_per = lambda_per

    def forward(self, imgA, imgB):
        """
        imgA, imgB: [B, C, H, W], range [0,1]
        """
        loss_l1 = self.l1(imgA, imgB)

        imgA_lpips, imgB_lpips = imgA * 2 - 1, imgB * 2 - 1
        loss_lpips = self.lpips(imgA_lpips, imgB_lpips).mean()

        loss_ssim = 1 - self.ssim(imgA, imgB)

        total_loss = loss_l1 + self.lambda_per * loss_lpips + self.lambda_ssim * loss_ssim
        return total_loss



if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms.functional as TF
    import torch

    device = torch.device("cuda")
    imgA = Image.open("D:/BaiduNetdiskDownload/fiveK/input/JPG/480p/a0001.jpg").convert("RGB")
    imgB = Image.open("D:/BaiduNetdiskDownload/fiveK/expertC/JPG/480p/a0001.jpg").convert("RGB")
    imgA = TF.to_tensor(imgA).unsqueeze(0).to(device)
    imgB = TF.to_tensor(imgB).unsqueeze(0).to(device)
    loss_fn = Loss(lambda_ssim=0.05, lambda_per=0.005, device=device)
    loss = loss_fn(imgA, imgB)
    print("loss: ", loss)
