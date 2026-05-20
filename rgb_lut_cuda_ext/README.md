# rgb_lut_cuda

CUDA trilinear RGB LUT sampling for tensors without a batch dimension:

- `img`: `[3, H, W]`
- `lut`: `[3, D, D, D]`
- output: `[3, H, W]`

The sampling order matches the original `grid_sample` implementation:

```python
grid = img.permute(1, 2, 0)[..., [2, 1, 0]]
grid = grid * 2 - 1
out = F.grid_sample(
    lut.unsqueeze(0),
    grid.unsqueeze(0).unsqueeze(0),
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
).squeeze(0).squeeze(1)
```

## Build

```bash
cd /Users/lihang/code/rgb_lut_cuda_ext
pip install -e .
```

or:

```bash
cd /Users/lihang/code/rgb_lut_cuda_ext
python setup.py install
```

## Use

```python
import torch
from rgb_lut_cuda import RGBLUTApply, apply_rgb_lut

img = torch.rand(3, 512, 512, device="cuda", requires_grad=True)
lut = torch.rand(3, 33, 33, 33, device="cuda", requires_grad=True)

out = apply_rgb_lut(img, lut)
loss = out.mean()
loss.backward()

module = RGBLUTApply()
out2 = module(img, lut)
```

## Check Correctness

```bash
cd /Users/lihang/code/rgb_lut_cuda_ext
python test_rgb_lut.py
```
