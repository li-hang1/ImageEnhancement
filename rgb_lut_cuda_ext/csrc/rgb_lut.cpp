#include <torch/extension.h>
#include <vector>

torch::Tensor rgb_lut_forward_cuda(torch::Tensor img, torch::Tensor lut);
std::vector<torch::Tensor> rgb_lut_backward_cuda(torch::Tensor grad_output, torch::Tensor img, torch::Tensor lut);

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_img_lut(const torch::Tensor& img, const torch::Tensor& lut) {
  check_cuda_contiguous(img, "img");
  check_cuda_contiguous(lut, "lut");

  TORCH_CHECK(img.dim() == 3, "img must have shape [3, H, W]");
  TORCH_CHECK(img.size(0) == 3, "img must have shape [3, H, W]");
  TORCH_CHECK(img.size(1) > 0 && img.size(2) > 0, "img H and W must be non-empty");

  TORCH_CHECK(lut.dim() == 4, "lut must have shape [3, D, D, D]");
  TORCH_CHECK(lut.size(0) == 3, "lut must have shape [3, D, D, D]");
  TORCH_CHECK(lut.size(1) > 0, "lut D must be non-empty");
  TORCH_CHECK(lut.size(1) == lut.size(2) && lut.size(1) == lut.size(3), "lut spatial dimensions must be equal");

  TORCH_CHECK(img.scalar_type() == lut.scalar_type(), "img and lut must have the same dtype");
  TORCH_CHECK(img.device() == lut.device(), "img and lut must be on the same CUDA device");
}

}  // namespace

torch::Tensor rgb_lut_forward(torch::Tensor img, torch::Tensor lut) {
  check_img_lut(img, lut);
  return rgb_lut_forward_cuda(img, lut);
}

std::vector<torch::Tensor> rgb_lut_backward(torch::Tensor grad_output,
                                            torch::Tensor img,
                                            torch::Tensor lut) {
  check_img_lut(img, lut);
  check_cuda_contiguous(grad_output, "grad_output");
  TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [3, H, W]");
  TORCH_CHECK(grad_output.size(0) == 3, "grad_output must have shape [3, H, W]");
  TORCH_CHECK(grad_output.size(1) == img.size(1) && grad_output.size(2) == img.size(2), "grad_output shape must match img shape");
  TORCH_CHECK(grad_output.scalar_type() == img.scalar_type(), "grad_output and img must have the same dtype");
  TORCH_CHECK(grad_output.device() == img.device(), "grad_output and img must be on the same CUDA device");
  return rgb_lut_backward_cuda(grad_output, img, lut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rgb_lut_forward, "RGB LUT trilinear forward (CUDA)");
  m.def("backward", &rgb_lut_backward, "RGB LUT trilinear backward (CUDA)");
}
