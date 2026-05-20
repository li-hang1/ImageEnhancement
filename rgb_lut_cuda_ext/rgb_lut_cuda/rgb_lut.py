import torch
import torch.nn.functional as F
from torch import nn

try:
    import _rgb_lut_cuda_ext
except ImportError as exc:
    _rgb_lut_cuda_ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _check_inputs(img: torch.Tensor, lut: torch.Tensor) -> None:
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError(f"img must have shape [3, H, W], got {tuple(img.shape)}")
    if lut.dim() != 4 or lut.size(0) != 3:
        raise ValueError(f"lut must have shape [3, D, D, D], got {tuple(lut.shape)}")
    if lut.size(1) != lut.size(2) or lut.size(1) != lut.size(3):
        raise ValueError(f"lut spatial dimensions must be equal, got {tuple(lut.shape)}")
    if img.device != lut.device:
        raise ValueError(f"img and lut must be on the same device, got {img.device} and {lut.device}")
    if img.dtype != lut.dtype:
        raise ValueError(f"img and lut must have the same dtype, got {img.dtype} and {lut.dtype}")
    if not img.is_cuda or not lut.is_cuda:
        raise ValueError("rgb_lut_cuda only supports CUDA tensors")


class _ApplyRGBLUTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        if _rgb_lut_cuda_ext is None:
            raise ImportError(
                "Failed to import _rgb_lut_cuda_ext. Build/install this package with "
                "`python setup.py install` or `pip install -e .` from rgb_lut_cuda_ext."
            ) from _IMPORT_ERROR
        _check_inputs(img, lut)
        img_contig = img.contiguous()
        lut_contig = lut.contiguous()
        output = _rgb_lut_cuda_ext.forward(img_contig, lut_contig)
        ctx.save_for_backward(img_contig, lut_contig)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        img, lut = ctx.saved_tensors
        grad_img, grad_lut = _rgb_lut_cuda_ext.backward(grad_output.contiguous(), img, lut)
        return grad_img, grad_lut


def apply_rgb_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """
    CUDA trilinear RGB LUT sampling with the same semantics as:

        grid = img.permute(1, 2, 0)[..., [2, 1, 0]]
        grid = grid * 2 - 1
        output = F.grid_sample(
            lut.unsqueeze(0), grid.unsqueeze(0).unsqueeze(0),
            mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(0).squeeze(1)

    Args:
        img: CUDA tensor [3, H, W], values usually in [0, 1].
        lut: CUDA tensor [3, D, D, D].

    Returns:
        CUDA tensor [3, H, W].
    """
    return _ApplyRGBLUTFunction.apply(img, lut)


def apply_rgb_lut_reference(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation used for correctness checks."""
    if img.dim() != 3 or lut.dim() != 4:
        raise ValueError("expected img [3,H,W] and lut [3,D,D,D]")
    grid = img.permute(1, 2, 0)[..., [2, 1, 0]]
    grid = grid * 2 - 1
    output = F.grid_sample(
        lut.unsqueeze(0),
        grid.unsqueeze(0).unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output.squeeze(0).squeeze(1)


class RGBLUTApply(nn.Module):
    """Small nn.Module wrapper for direct insertion into a network."""

    def forward(self, img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        return apply_rgb_lut(img, lut)
