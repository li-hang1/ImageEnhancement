#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>

namespace {

constexpr int kThreads = 256;

// D 是网格中一条棱的顶点数
// 内部小函数适合用 __forceinline__
template <typename scalar_t>
__device__ __forceinline__ int64_t lut_index(int64_t c, int64_t z, int64_t y, int64_t x, int64_t D) {
  return ((c * D + z) * D + y) * D + x;  // c*D^3 + z*D^2 + y*D + x
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t clip_coordinate(scalar_t in, int64_t D) {
  if (isnan(static_cast<double>(in))) {
    return static_cast<scalar_t>(0);
  }
  const scalar_t max_v = static_cast<scalar_t>(D - 1);
  if (in <= static_cast<scalar_t>(0)) {
    return static_cast<scalar_t>(0);
  }
  if (in >= max_v) {
    return max_v;
  }
  return in;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t clip_coordinate_set_grad(scalar_t in,
                                                            int64_t D,
                                                            scalar_t* grad_in) {
  if (isnan(static_cast<double>(in))) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  const scalar_t max_v = static_cast<scalar_t>(D - 1);
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  if (in >= max_v) {
    *grad_in = static_cast<scalar_t>(0);
    return max_v;
  }
  *grad_in = static_cast<scalar_t>(1);
  return in;
}

template <typename scalar_t>
__device__ __forceinline__ void compute_position(scalar_t img_value,
                                                 int64_t D,
                                                 scalar_t* coord,
                                                 int64_t* i0,
                                                 int64_t* i1,
                                                 scalar_t* w0,
                                                 scalar_t* w1) {
  const scalar_t scale = static_cast<scalar_t>(D - 1);
  const scalar_t clipped = clip_coordinate(img_value * scale, D);
  const int64_t base = static_cast<int64_t>(floor(static_cast<double>(clipped)));
  const int64_t upper = base + 1 < D ? base + 1 : D - 1;
  const scalar_t t = clipped - static_cast<scalar_t>(base);

  *coord = clipped;
  *i0 = base;
  *i1 = upper;
  *w0 = static_cast<scalar_t>(1) - t;
  *w1 = t;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lut_value(const scalar_t* __restrict__ lut,
                                             int64_t c,
                                             int64_t z,
                                             int64_t y,
                                             int64_t x,
                                             int64_t D) {
  return lut[lut_index<scalar_t>(c, z, y, x, D)];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t interpolate_channel(const scalar_t* __restrict__ lut,
                                                       int64_t c,
                                                       int64_t z0,
                                                       int64_t z1,
                                                       int64_t y0,
                                                       int64_t y1,
                                                       int64_t x0,
                                                       int64_t x1,
                                                       scalar_t wz0,
                                                       scalar_t wz1,
                                                       scalar_t wy0,
                                                       scalar_t wy1,
                                                       scalar_t wx0,
                                                       scalar_t wx1,
                                                       int64_t D) {
  const scalar_t v000 = lut_value(lut, c, z0, y0, x0, D);
  const scalar_t v001 = lut_value(lut, c, z0, y0, x1, D);
  const scalar_t v010 = lut_value(lut, c, z0, y1, x0, D);
  const scalar_t v011 = lut_value(lut, c, z0, y1, x1, D);
  const scalar_t v100 = lut_value(lut, c, z1, y0, x0, D);
  const scalar_t v101 = lut_value(lut, c, z1, y0, x1, D);
  const scalar_t v110 = lut_value(lut, c, z1, y1, x0, D);
  const scalar_t v111 = lut_value(lut, c, z1, y1, x1, D);

  return v000 * wz0 * wy0 * wx0 +
         v001 * wz0 * wy0 * wx1 +
         v010 * wz0 * wy1 * wx0 +
         v011 * wz0 * wy1 * wx1 +
         v100 * wz1 * wy0 * wx0 +
         v101 * wz1 * wy0 * wx1 +
         v110 * wz1 * wy1 * wx0 +
         v111 * wz1 * wy1 * wx1;
}

template <typename scalar_t>
__global__ void rgb_lut_forward_kernel(const scalar_t* __restrict__ img,
                                       const scalar_t* __restrict__ lut,
                                       scalar_t* __restrict__ output,
                                       int64_t H,
                                       int64_t W,
                                       int64_t D,
                                       int64_t pixels) {
  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pixels) {
    return;
  }

  const int64_t hw = H * W;
  const scalar_t r = img[0 * hw + p];
  const scalar_t g = img[1 * hw + p];
  const scalar_t b = img[2 * hw + p];

  scalar_t x, y, z;
  int64_t x0, x1, y0, y1, z0, z1;
  scalar_t wx0, wx1, wy0, wy1, wz0, wz1;

  compute_position(b, D, &x, &x0, &x1, &wx0, &wx1);
  compute_position(g, D, &y, &y0, &y1, &wy0, &wy1);
  compute_position(r, D, &z, &z0, &z1, &wz0, &wz1);

  for (int64_t c = 0; c < 3; ++c) {
    output[c * hw + p] = interpolate_channel(lut, c, z0, z1, y0, y1, x0, x1, wz0, wz1, wy0, wy1, wx0, wx1, D);
  }
}

template <typename scalar_t>
__device__ __forceinline__ void add_lut_grad(scalar_t* __restrict__ grad_lut,
                                             int64_t c,
                                             int64_t z,
                                             int64_t y,
                                             int64_t x,
                                             int64_t D,
                                             scalar_t value) {
  atomicAdd(&grad_lut[lut_index<scalar_t>(c, z, y, x, D)], value);
}

template <typename scalar_t>
__global__ void rgb_lut_backward_kernel(const scalar_t* __restrict__ grad_output,
                                        const scalar_t* __restrict__ img,
                                        const scalar_t* __restrict__ lut,
                                        scalar_t* __restrict__ grad_img,
                                        scalar_t* __restrict__ grad_lut,
                                        int64_t H,
                                        int64_t W,
                                        int64_t D,
                                        int64_t pixels) {
  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pixels) {
    return;
  }

  const int64_t hw = H * W;
  const scalar_t r = img[0 * hw + p];
  const scalar_t g = img[1 * hw + p];
  const scalar_t b = img[2 * hw + p];
  const scalar_t scale = static_cast<scalar_t>(D - 1);

  scalar_t grad_x_clip, grad_y_clip, grad_z_clip;
  const scalar_t x = clip_coordinate_set_grad(b * scale, D, &grad_x_clip);
  const scalar_t y = clip_coordinate_set_grad(g * scale, D, &grad_y_clip);
  const scalar_t z = clip_coordinate_set_grad(r * scale, D, &grad_z_clip);

  const int64_t x0 = static_cast<int64_t>(floor(static_cast<double>(x)));
  const int64_t y0 = static_cast<int64_t>(floor(static_cast<double>(y)));
  const int64_t z0 = static_cast<int64_t>(floor(static_cast<double>(z)));
  const int64_t x1 = x0 + 1 < D ? x0 + 1 : D - 1;
  const int64_t y1 = y0 + 1 < D ? y0 + 1 : D - 1;
  const int64_t z1 = z0 + 1 < D ? z0 + 1 : D - 1;

  const scalar_t wx1 = x - static_cast<scalar_t>(x0);
  const scalar_t wy1 = y - static_cast<scalar_t>(y0);
  const scalar_t wz1 = z - static_cast<scalar_t>(z0);
  const scalar_t wx0 = static_cast<scalar_t>(1) - wx1;
  const scalar_t wy0 = static_cast<scalar_t>(1) - wy1;
  const scalar_t wz0 = static_cast<scalar_t>(1) - wz1;

  scalar_t grad_x = static_cast<scalar_t>(0);
  scalar_t grad_y = static_cast<scalar_t>(0);
  scalar_t grad_z = static_cast<scalar_t>(0);

  for (int64_t c = 0; c < 3; ++c) {
    const scalar_t go = grad_output[c * hw + p];  // 每个变量只跟输出向量中的三个变量有关系，所以只取同像素位置的三个外层梯度
    // 图像的梯度，雅可比行列式是五条带状对角线

    add_lut_grad(grad_lut, c, z0, y0, x0, D, go * wz0 * wy0 * wx0);
    add_lut_grad(grad_lut, c, z0, y0, x1, D, go * wz0 * wy0 * wx1);
    add_lut_grad(grad_lut, c, z0, y1, x0, D, go * wz0 * wy1 * wx0);
    add_lut_grad(grad_lut, c, z0, y1, x1, D, go * wz0 * wy1 * wx1);
    add_lut_grad(grad_lut, c, z1, y0, x0, D, go * wz1 * wy0 * wx0);
    add_lut_grad(grad_lut, c, z1, y0, x1, D, go * wz1 * wy0 * wx1);
    add_lut_grad(grad_lut, c, z1, y1, x0, D, go * wz1 * wy1 * wx0);
    add_lut_grad(grad_lut, c, z1, y1, x1, D, go * wz1 * wy1 * wx1);

    const scalar_t v000 = lut_value(lut, c, z0, y0, x0, D);
    const scalar_t v001 = lut_value(lut, c, z0, y0, x1, D);
    const scalar_t v010 = lut_value(lut, c, z0, y1, x0, D);
    const scalar_t v011 = lut_value(lut, c, z0, y1, x1, D);
    const scalar_t v100 = lut_value(lut, c, z1, y0, x0, D);
    const scalar_t v101 = lut_value(lut, c, z1, y0, x1, D);
    const scalar_t v110 = lut_value(lut, c, z1, y1, x0, D);
    const scalar_t v111 = lut_value(lut, c, z1, y1, x1, D);

    const scalar_t d_out_dx =
        (v001 - v000) * wz0 * wy0 +
        (v011 - v010) * wz0 * wy1 +
        (v101 - v100) * wz1 * wy0 +
        (v111 - v110) * wz1 * wy1;

    const scalar_t d_out_dy =
        (v010 - v000) * wz0 * wx0 +
        (v011 - v001) * wz0 * wx1 +
        (v110 - v100) * wz1 * wx0 +
        (v111 - v101) * wz1 * wx1;

    const scalar_t d_out_dz =
        (v100 - v000) * wy0 * wx0 +
        (v101 - v001) * wy0 * wx1 +
        (v110 - v010) * wy1 * wx0 +
        (v111 - v011) * wy1 * wx1;

    grad_x += go * d_out_dx;
    grad_y += go * d_out_dy;
    grad_z += go * d_out_dz;
  }

  grad_img[0 * hw + p] = grad_z * grad_z_clip * scale;  // floor的导数为0
  grad_img[1 * hw + p] = grad_y * grad_y_clip * scale;
  grad_img[2 * hw + p] = grad_x * grad_x_clip * scale;
}

}  // namespace

torch::Tensor rgb_lut_forward_cuda(torch::Tensor img, torch::Tensor lut) {
  const c10::cuda::CUDAGuard device_guard(img.device());

  const int64_t H = img.size(1);
  const int64_t W = img.size(2);
  const int64_t D = lut.size(1);
  const int64_t pixels = H * W;
  auto output = torch::empty_like(img);

  const dim3 block(kThreads);
  const dim3 grid((pixels + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "rgb_lut_forward_cuda", [&] {
    rgb_lut_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
        img.data_ptr<scalar_t>(),
        lut.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        H,
        W,
        D,
        pixels);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> rgb_lut_backward_cuda(torch::Tensor grad_output, torch::Tensor img, torch::Tensor lut) {
  const c10::cuda::CUDAGuard device_guard(img.device());

  const int64_t H = img.size(1);
  const int64_t W = img.size(2);
  const int64_t D = lut.size(1);
  const int64_t pixels = H * W;
  auto grad_img = torch::empty_like(img);
  auto grad_lut = torch::zeros_like(lut);

  const dim3 block(kThreads);
  const dim3 grid((pixels + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "rgb_lut_backward_cuda", [&] {
    rgb_lut_backward_kernel<scalar_t><<<grid, block, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        img.data_ptr<scalar_t>(),
        lut.data_ptr<scalar_t>(),
        grad_img.data_ptr<scalar_t>(),
        grad_lut.data_ptr<scalar_t>(),
        H,
        W,
        D,
        pixels);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_img, grad_lut};
}
