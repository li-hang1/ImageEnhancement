import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/root/LH/LoRALUT/rgb_lut_cuda_ext")
from rgb_lut_cuda import apply_rgb_lut


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, downsample_factor=8):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.layers(x)


def bilinear_interpolate_feature(feat_map, original_h, original_w):
    """
    根据原图每个像素坐标，在特征图上按比例插值取值
    最终输出和原图一样大的特征图
    
    参数：
        feat_map: 特征图 [B, C, H, W]
        original_h: 原图高度
        original_w: 原图宽度
    返回：
        aligned_feat: [B, C, original_h, original_w]
    """
    B, C, H_feat, W_feat = feat_map.shape
    device = feat_map.device

    # 生成归一化的 x, y 坐标（grid_sample 顺序是 x, y）
    y = torch.linspace(-1, 1, original_h, device=device)  # 高度方向
    x = torch.linspace(-1, 1, original_w, device=device)  # 宽度方向
    
    # 生成网格 [original_h, original_w, 2]
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

    # align_corners=True：保证坐标严格对齐
    aligned_feat = F.grid_sample(
        feat_map,       # 输入特征图
        grid,           # 目标坐标网格
        mode='bilinear',# 双线性插值（自动用周围4个像素加权）
        padding_mode='zeros',
        align_corners=True  # 关键：坐标严格按比例映射
    )

    return aligned_feat


class LoraLUT(nn.Module):
    def __init__(self):
        super().__init__()
        self.Upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),   # [B, 8, 128, 128]
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),   # [B, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [B, 64, 8, 8]
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, 64)
        self.feat_weight_generator = nn.Linear(64, 64)
        
        self.extractor = FeatureExtractor(in_channels=3, out_channels=64, downsample_factor=8)

        self.rgb1 = nn.Linear(64, 3 * 3 * 16)
        self.rgb2 = nn.Linear(64, 3 * 3 * 16)
        self.rgb3 = nn.Linear(64, 3 * 3 * 16)

        self.rxy1 = nn.Linear(64, 3 * 3 * 16)
        self.rxy2 = nn.Linear(64, 3 * 3 * 16)
        self.rxy3 = nn.Linear(64, 3 * 3 * 16)

        self.gxy1 = nn.Linear(64, 3 * 3 * 16)
        self.gxy2 = nn.Linear(64, 3 * 3 * 16)
        self.gxy3 = nn.Linear(64, 3 * 3 * 16)

        self.bxy1 = nn.Linear(64, 3 * 3 * 16)
        self.bxy2 = nn.Linear(64, 3 * 3 * 16)
        self.bxy3 = nn.Linear(64, 3 * 3 * 16)

        lut_layers = [
            self.rgb1, self.rgb2, self.rgb3,
            self.rxy1, self.rxy2, self.rxy3,
            self.gxy1, self.gxy2, self.gxy3,
            self.bxy1, self.bxy2, self.bxy3,
        ]

        for layer in lut_layers:
            layer.weight.data *= 0.001
            layer.bias.data *= 0.001

        self.apply_rgb_lut = apply_rgb_lut


    def forward(self, img):
        x = self.Upsample(img)
        x = self.cnn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        
        feat = self.extractor(img)  # [B, feat_dim, H_feat, W_heat]
        
        feat_weight = self.feat_weight_generator(x)  # [B, feat_dim]
        
        feat = feat * feat_weight[:, :, None, None]
        feat = feat.sum(dim = 1, keepdim = True)  # [B, 1, H_feat, W_heat]
        
        aligned_feat = bilinear_interpolate_feature(feat, img.shape[2], img.shape[3])  # [B, 1, H_img, W_img]

        rgb1 = self.rgb1(F.relu(x)).reshape(3, 3, 16)
        rgb2 = self.rgb2(F.relu(x)).reshape(3, 3, 16)
        rgb3 = self.rgb3(F.relu(x)).reshape(3, 3, 16)

        rxy1 = self.rxy1(F.relu(x)).reshape(3, 3, 16)
        rxy2 = self.rxy2(F.relu(x)).reshape(3, 3, 16)
        rxy3 = self.rxy3(F.relu(x)).reshape(3, 3, 16)

        gxy1 = self.gxy1(F.relu(x)).reshape(3, 3, 16)
        gxy2 = self.gxy2(F.relu(x)).reshape(3, 3, 16)
        gxy3 = self.gxy3(F.relu(x)).reshape(3, 3, 16)

        bxy1 = self.bxy1(F.relu(x)).reshape(3, 3, 16)
        bxy2 = self.bxy2(F.relu(x)).reshape(3, 3, 16)
        bxy3 = self.bxy3(F.relu(x)).reshape(3, 3, 16)

        rgb_lut = sum(torch.einsum('bi,bj,bk->bijk', rgb1[i], rgb2[i], rgb3[i]).unsqueeze(0) for i in range(3))
        rxy_lut = sum(torch.einsum('bi,bj,bk->bijk', rxy1[i], rxy2[i], rxy3[i]).unsqueeze(0) for i in range(3))
        gxy_lut = sum(torch.einsum('bi,bj,bk->bijk', gxy1[i], gxy2[i], gxy3[i]).unsqueeze(0) for i in range(3))
        bxy_lut = sum(torch.einsum('bi,bj,bk->bijk', bxy1[i], bxy2[i], bxy3[i]).unsqueeze(0) for i in range(3))

        rgb_output = self.apply_rgb_lut(img.squeeze(0), rgb_lut.squeeze(0))
        rxy_output = self.apply_rgb_lut(torch.cat([img[:, [1, 2], :, :], aligned_feat], dim=1).squeeze(0), rxy_lut.squeeze(0))
        gxy_output = self.apply_rgb_lut(torch.cat([img[:, [0, 2], :, :], aligned_feat], dim=1).squeeze(0), gxy_lut.squeeze(0))
        bxy_output = self.apply_rgb_lut(torch.cat([img[:, [0, 1], :, :], aligned_feat], dim=1).squeeze(0), bxy_lut.squeeze(0))
        
        return img + rgb_output.unsqueeze(0) + rxy_output.unsqueeze(0) + gxy_output.unsqueeze(0) + bxy_output.unsqueeze(0)









if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.utils import save_image
    image = Image.open("/mnt/cloud_disk/fiveK/expertC/JPG/480p/a0001.jpg").convert("RGB")
    img = TF.to_tensor(image).unsqueeze(0).cuda()
    fake_target = torch.rand((1, 3, 480, 722)).cuda()

    model = LoraLUT().cuda()

    output = model(img)
    loss = F.mse_loss(output, fake_target)
    loss.backward()

    print("原图形状：", img.shape)
    print("输出图形状：", output.shape)
    print(f"loss: {loss.item():.4f}")
