#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename T>
struct Pos {
  int i0, i1;
  T w0, w1;
  T grad_clip;
};

template <typename T>
__device__ __forceinline__ T lerp(T a, T b, T t) {
  return fma(t, b - a, a);  // Fused Multiply-Add，融合乘加，std::fma(a, b, c)，作用：a * b + c
}

template <>
__device__ __forceinline__ float lerp<float>(float a, float b, float t) {
  return fmaf(t, b - a, a);
}

template <typename T>
__device__ __forceinline__ Pos<T> compute_pos(T v, int D) {
  const T scale = static_cast<T>(D - 1);
  T x = v * scale;
  T grad = static_cast<T>(1);

  if (x != x) {
    // 用 x!=x 判断x是不是nan
    x = static_cast<T>(0);
    grad = static_cast<T>(0);
  } else if (x <= static_cast<T>(0)) {
    x = static_cast<T>(0);
    grad = static_cast<T>(0);
  } else if (x >= scale) {
    x = scale;
    grad = static_cast<T>(0);
  }

  const int i0 = static_cast<int>(x);  // x 已经 >= 0，等价 floor(x)
  const int i1 = i0 + 1 < D ? i0 + 1 : D - 1;
  const T w1 = x - static_cast<T>(i0);

  return {i0, i1, static_cast<T>(1) - w1, w1, grad};
}

template <typename T>
__device__ __forceinline__ void make_offsets(int D, const Pos<T>& x, const Pos<T>& y, const Pos<T>& z,
                                             int& o000, int& o001, int& o010, int& o011,
                                             int& o100, int& o101, int& o110, int& o111) {
  const int D2 = D * D;
  const int z0 = z.i0 * D2;
  const int z1 = z.i1 * D2;
  const int y0 = y.i0 * D;
  const int y1 = y.i1 * D;

  o000 = z0 + y0 + x.i0;
  o001 = z0 + y0 + x.i1;
  o010 = z0 + y1 + x.i0;
  o011 = z0 + y1 + x.i1;
  o100 = z1 + y0 + x.i0;
  o101 = z1 + y0 + x.i1;
  o110 = z1 + y1 + x.i0;
  o111 = z1 + y1 + x.i1;
}

template <typename T>
__device__ __forceinline__ T trilerp(const T* __restrict__ l,
                                     int o000, int o001, int o010, int o011,
                                     int o100, int o101, int o110, int o111,
                                     T tx, T ty, T tz) {
  const T v000 = l[o000], v001 = l[o001], v010 = l[o010], v011 = l[o011];
  const T v100 = l[o100], v101 = l[o101], v110 = l[o110], v111 = l[o111];

  const T c00 = lerp(v000, v001, tx);
  const T c01 = lerp(v010, v011, tx);
  const T c10 = lerp(v100, v101, tx);
  const T c11 = lerp(v110, v111, tx);

  const T c0 = lerp(c00, c01, ty);
  const T c1 = lerp(c10, c11, ty);
  return lerp(c0, c1, tz);
}

template <typename T>
__device__ __forceinline__ void atomic_add_fast(T* addr, T v) {
  // 反向传播时使用，addr是地址，v是要累加的值
  if (v == static_cast<T>(0)) return;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const unsigned active = __activemask();  // 返回一个 32-bit bitmask（位掩码），表示当前 warp 里哪些线程是“active（活跃的）”
  const unsigned long long key = reinterpret_cast<unsigned long long>(addr);
  // __match_any_sync(active, key)是束内匹配函数，用于warp中相同key线程执行操作
  const unsigned peers = __match_any_sync(active, key);  // 返回32bit无符号整数，只有active中活跃的才执行这一行
  const int lane = threadIdx.x & 31;  // 31的二进制是11111
  const int leader = __ffs(peers) - 1;  // find first set bit，找到 x 中从低位开始第一个为 1 的 bit，并返回它的“位位置（从 1 开始计数）”，

  T sum = static_cast<T>(0);
  unsigned m = peers;
  while (m) {
    const int src = __ffs(m) - 1;  // 取还没算到的那一位
    sum += __shfl_sync(active, v, src);  // 把那位的值加进sum
    m &= m - 1;  // 只消掉最低位的那个 1
  }

  if (lane == leader) atomicAdd(addr, sum);
#else
  atomicAdd(addr, v);
#endif
}

template <typename T>
__global__ void rgb_lut_forward_kernel_fast(const T* __restrict__ img,
                                            const T* __restrict__ lut,
                                            T* __restrict__ out,
                                            int H, int W, int D, int pixels) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;

  const int hw = H * W;
  const T r = img[p];
  const T g = img[hw + p];
  const T b = img[2 * hw + p];

  const Pos<T> x = compute_pos(b, D);
  const Pos<T> y = compute_pos(g, D);
  const Pos<T> z = compute_pos(r, D);

  int o000, o001, o010, o011, o100, o101, o110, o111;
  make_offsets(D, x, y, z, o000, o001, o010, o011, o100, o101, o110, o111);

  const int D3 = D * D * D;

#pragma unroll
  for (int c = 0; c < 3; ++c) {
    out[c * hw + p] = trilerp(
        lut + c * D3,
        o000, o001, o010, o011, o100, o101, o110, o111,
        x.w1, y.w1, z.w1);
  }
}

template <typename T>
__global__ void rgb_lut_backward_kernel_fast(const T* __restrict__ grad_out,
                                             const T* __restrict__ img,
                                             const T* __restrict__ lut,
                                             T* __restrict__ grad_img,
                                             T* __restrict__ grad_lut,
                                             int H, int W, int D, int pixels) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;

  const int hw = H * W;
  const int D3 = D * D * D;
  const T scale = static_cast<T>(D - 1);

  const Pos<T> x = compute_pos(img[2 * hw + p], D);  // b -> x
  const Pos<T> y = compute_pos(img[hw + p], D);      // g -> y
  const Pos<T> z = compute_pos(img[p], D);           // r -> z

  int o000, o001, o010, o011, o100, o101, o110, o111;
  make_offsets(D, x, y, z, o000, o001, o010, o011, o100, o101, o110, o111);

  const T yz00 = z.w0 * y.w0, yz01 = z.w0 * y.w1;
  const T yz10 = z.w1 * y.w0, yz11 = z.w1 * y.w1;
  const T zx00 = z.w0 * x.w0, zx01 = z.w0 * x.w1;
  const T zx10 = z.w1 * x.w0, zx11 = z.w1 * x.w1;
  const T yx00 = y.w0 * x.w0, yx01 = y.w0 * x.w1;
  const T yx10 = y.w1 * x.w0, yx11 = y.w1 * x.w1;

  const T w000 = yz00 * x.w0, w001 = yz00 * x.w1;
  const T w010 = yz01 * x.w0, w011 = yz01 * x.w1;
  const T w100 = yz10 * x.w0, w101 = yz10 * x.w1;
  const T w110 = yz11 * x.w0, w111 = yz11 * x.w1;

  T gx = static_cast<T>(0);
  T gy = static_cast<T>(0);
  T gz = static_cast<T>(0);

#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const T go = grad_out[c * hw + p];
    T* gl = grad_lut + c * D3;

    atomic_add_fast(gl + o000, go * w000);
    atomic_add_fast(gl + o001, go * w001);
    atomic_add_fast(gl + o010, go * w010);
    atomic_add_fast(gl + o011, go * w011);
    atomic_add_fast(gl + o100, go * w100);
    atomic_add_fast(gl + o101, go * w101);
    atomic_add_fast(gl + o110, go * w110);
    atomic_add_fast(gl + o111, go * w111);

    const T* l = lut + c * D3;
    const T v000 = l[o000], v001 = l[o001], v010 = l[o010], v011 = l[o011];
    const T v100 = l[o100], v101 = l[o101], v110 = l[o110], v111 = l[o111];

    gx += go * ((v001 - v000) * yz00 + (v011 - v010) * yz01 +
                (v101 - v100) * yz10 + (v111 - v110) * yz11);

    gy += go * ((v010 - v000) * zx00 + (v011 - v001) * zx01 +
                (v110 - v100) * zx10 + (v111 - v101) * zx11);

    gz += go * ((v100 - v000) * yx00 + (v101 - v001) * yx01 +
                (v110 - v010) * yx10 + (v111 - v011) * yx11);
  }

  grad_img[p] = gz * z.grad_clip * scale;
  grad_img[hw + p] = gy * y.grad_clip * scale;
  grad_img[2 * hw + p] = gx * x.grad_clip * scale;
}

}  // namespace


torch::Tensor rgb_lut_forward_cuda(torch::Tensor img, torch::Tensor lut) {
  const c10::cuda::CUDAGuard device_guard(img.device());

  const int64_t pixels64 = img.size(1) * img.size(2);  // 图像大小
  TORCH_CHECK(pixels64 <= INT_MAX, "too many pixels for int indexing");
  TORCH_CHECK(lut.numel() <= INT_MAX, "lut too large for int indexing");

  const int H = static_cast<int>(img.size(1));
  const int W = static_cast<int>(img.size(2));
  const int D = static_cast<int>(lut.size(1));
  const int pixels = static_cast<int>(pixels64);

  auto output = torch::empty_like(img);
  const dim3 block(kThreads);
  const dim3 grid((pixels + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "rgb_lut_forward_cuda_fast", [&] {
    rgb_lut_forward_kernel_fast<scalar_t><<<grid, block, 0, stream>>>(
        img.data_ptr<scalar_t>(), lut.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), H, W, D, pixels);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> rgb_lut_backward_cuda(torch::Tensor grad_output, torch::Tensor img, torch::Tensor lut) {
  const c10::cuda::CUDAGuard device_guard(img.device());

  const int64_t pixels64 = img.size(1) * img.size(2);
  TORCH_CHECK(pixels64 <= INT_MAX, "too many pixels for int indexing");
  TORCH_CHECK(lut.numel() <= INT_MAX, "lut too large for int indexing");

  const int H = static_cast<int>(img.size(1));
  const int W = static_cast<int>(img.size(2));
  const int D = static_cast<int>(lut.size(1));
  const int pixels = static_cast<int>(pixels64);

  auto grad_img = torch::empty_like(img);
  auto grad_lut = torch::zeros_like(lut);

  const dim3 block(kThreads);
  const dim3 grid((pixels + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "rgb_lut_backward_cuda_fast", [&] {
    rgb_lut_backward_kernel_fast<scalar_t><<<grid, block, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(), img.data_ptr<scalar_t>(),
        lut.data_ptr<scalar_t>(), grad_img.data_ptr<scalar_t>(),
        grad_lut.data_ptr<scalar_t>(), H, W, D, pixels);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_img, grad_lut};
}
