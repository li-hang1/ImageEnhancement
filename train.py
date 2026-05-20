import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from datetime import datetime
from torchvision.utils import save_image

from loralut_dataset import ImageDatasetFiveK
from model import LoraLUT
from loss import Loss


root_dir = "./checkpoints_cuda_lut"
os.makedirs(root_dir, exist_ok=True)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(root_dir, f"exp_{time_str}")
os.makedirs(exp_dir, exist_ok=True)

log_path = os.path.join(exp_dir, "log.txt")

model_dir = os.path.join(exp_dir, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_epoch_{}.pth")

image_dir = os.path.join(exp_dir, "images")
os.makedirs(image_dir, exist_ok=True)

with open(log_path, "w") as f:
    f.write("epoch, train_loss, train_psnr, val_psnr\n")

device = torch.device("cuda")

model = LoraLUT().to(device)
# model.load_state_dict(torch.load("checkpoints_cuda_lut/exp_20260509_170235/model/model_epoch_169.pth"))
total_params = sum(p.numel() for p in model.parameters())
print(f"total_params: {total_params}")

train_dataset = ImageDatasetFiveK("/mnt/cloud_disk/fiveK", mode="train")
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = ImageDatasetFiveK("/mnt/cloud_disk/fiveK", mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loss_fn = Loss(lambda_ssim=0.0, lambda_per=0.0, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

psnr_metric = PeakSignalNoiseRatio(data_range=(0, 1)).to(device)

for epoch in range(1000):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for i, data in enumerate(train_dataloader):
            img = data["input"].to(device)
            target = data["exptC"].to(device)

            optimizer.zero_grad()
            output = model(img)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            psnr = psnr_metric(output, target)

            running_loss += loss.item()
            running_psnr += psnr.item()

            avg_loss = running_loss / (i + 1)
            avg_psnr = running_psnr / (i + 1)

            pbar.set_postfix(loss=f"{avg_loss:.4f}", psnr=f"{avg_psnr:.4f}")
            pbar.update()

    epoch_loss = running_loss / len(train_dataloader)
    train_psnr = running_psnr / len(train_dataloader)

    model.eval()
    val_psnr = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            img = data["input"].to(device)
            target = data["exptC"].to(device)
            img_name = data["img_name"]

            output = model(img)

            psnr = psnr_metric(output, target)
            val_psnr += psnr.item()

            if i < 10:
                os.makedirs(os.path.join(image_dir, f"epoch_{epoch}"), exist_ok=True)
                save_image(output.clamp(0, 1), os.path.join(image_dir, f"epoch_{epoch}", f"{img_name[0]}.jpg"))

    val_psnr /= len(test_dataloader)

    print( f"[Epoch {epoch + 1}] Train Loss: {epoch_loss:.4f}, Train PSNR: {train_psnr:.4f}, Val PSNR: {val_psnr:.4f}")

    torch.save(model.state_dict(), model_path.format(epoch + 1))

    with open(log_path, "a") as f:
        f.write(f"{epoch + 1}, {epoch_loss:.6f}, {train_psnr:.4f}, {val_psnr:.4f}\n")




