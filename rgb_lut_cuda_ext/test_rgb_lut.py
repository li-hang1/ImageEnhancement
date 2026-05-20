import torch

from rgb_lut_cuda import apply_rgb_lut, apply_rgb_lut_reference


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    torch.manual_seed(7)
    device = "cuda"
    dtype = torch.float32
    H, W, D = 17, 19, 33

    img_a = torch.rand(3, H, W, device=device, dtype=dtype, requires_grad=True)
    lut_a = torch.rand(3, D, D, D, device=device, dtype=dtype, requires_grad=True)
    img_b = img_a.detach().clone().requires_grad_(True)
    lut_b = lut_a.detach().clone().requires_grad_(True)

    out_cuda = apply_rgb_lut(img_a, lut_a)
    out_ref = apply_rgb_lut_reference(img_b, lut_b)

    grad = torch.randn_like(out_cuda)
    out_cuda.backward(grad)
    out_ref.backward(grad)

    print("forward max abs diff:", (out_cuda - out_ref).abs().max().item())
    print("grad img max abs diff:", (img_a.grad - img_b.grad).abs().max().item())
    print("grad lut max abs diff:", (lut_a.grad - lut_b.grad).abs().max().item())

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(img_a.grad, img_b.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lut_a.grad, lut_b.grad, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    main()
