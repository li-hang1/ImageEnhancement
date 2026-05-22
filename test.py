from PIL import Image
import os
import torchvision.transforms.functional as TF
from model_cuda_lut import LoraLUT
import torch
from torchvision.utils import save_image


model_path = "./checkpoints_cuda_lut/exp_20260521_160822/model/model_epoch_197.pth"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda:0")
model = LoraLUT().to(device)
model.load_state_dict(torch.load(model_path))


for image_name in os.listdir("./demo"):
    image_path = os.path.join("./demo", image_name)
    image = Image.open(image_path).convert("RGB")
    image_tensor = TF.to_tensor(image).to(device).unsqueeze(0)
    output = model(image_tensor)
    save_image(output.clamp(0, 1), os.path.join(output_dir, image_name))
    


