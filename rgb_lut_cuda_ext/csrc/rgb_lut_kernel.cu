// #include <cuda_runtime.h>
// #include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <vector>

// namespace {

// constexpr int kThreads = 256;

// using acc_t = float;

// __device__ __forceinline__ float clip(float x, float max_v) {
//     return fminf(fmaxf(x, 0.f), max_v);
// }

// __device__ __forceinline__ int lut_idx(int c, int z, int y, int x, int D) {
//     return ((c * D + z) * D + y) * D + x;
// }

// template<typename scalar_t>
// __device__ __forceinline__ void compute_coord(scalar_t v, int D, int& i0, int& i1, float& w0, float& w1) {
//     float x = clip((float)v * (D - 1), (float)(D - 1));

//     i0 = (int)x;
//     i1 = min(i0 + 1, D - 1);

//     w1 = x - i0;
//     w0 = 1.f - w1;
// }

// template<typename scalar_t>
// __global__ void rgb_lut_forward_kernel(const scalar_t* __restrict__ img,
//                                        const scalar_t* __restrict__ lut,
//                                        scalar_t* __restrict__ out,
//                                        int H,
//                                        int W,
//                                        int D,
//                                        int pixels) {
    
//     int p = blockIdx.x * blockDim.x + threadIdx.x;

//     if (p >= pixels) return;

//     int hw = H * W;

//     float r = (float)img[p];
//     float g = (float)img[hw + p];
//     float b = (float)img[2 * hw + p];

//     int x0, x1, y0, y1, z0, z1;
//     float wx0, wx1, wy0, wy1, wz0, wz1;

//     compute_coord(b, D, x0, x1, wx0, wx1);
//     compute_coord(g, D, y0, y1, wy0, wy1);
//     compute_coord(r, D, z0, z1, wz0, wz1);

//     float w000 = wz0 * wy0 * wx0;
//     float w001 = wz0 * wy0 * wx1;
//     float w010 = wz0 * wy1 * wx0;
//     float w011 = wz0 * wy1 * wx1;

//     float w100 = wz1 * wy0 * wx0;
//     float w101 = wz1 * wy0 * wx1;
//     float w110 = wz1 * wy1 * wx0;
//     float w111 = wz1 * wy1 * wx1;

//     int idx000 = lut_idx(0, z0, y0, x0, D);
//     int idx001 = lut_idx(0, z0, y0, x1, D);
//     int idx010 = lut_idx(0, z0, y1, x0, D);
//     int idx011 = lut_idx(0, z0, y1, x1, D);

//     int idx100 = lut_idx(0, z1, y0, x0, D);
//     int idx101 = lut_idx(0, z1, y0, x1, D);
//     int idx110 = lut_idx(0, z1, y1, x0, D);
//     int idx111 = lut_idx(0, z1, y1, x1, D);

//     int stride_c = D * D * D;

//     #pragma unroll
//     for (int c = 0; c < 3; c++) {

//         const scalar_t* lut_c = lut + c * stride_c;

//         float v =
//             lut_c[idx000] * w000 +
//             lut_c[idx001] * w001 +
//             lut_c[idx010] * w010 +
//             lut_c[idx011] * w011 +
//             lut_c[idx100] * w100 +
//             lut_c[idx101] * w101 +
//             lut_c[idx110] * w110 +
//             lut_c[idx111] * w111;

//         out[c * hw + p] = (scalar_t)v;
//     }
// }

// // template<typename scalar_t>
// // __global__ void rgb_lut_backward_kernel(const scalar_t* __restrict__ grad_out,
// //                                         const scalar_t* __restrict__ img,
// //                                         const scalar_t* __restrict__ lut,
// //                                         scalar_t* __restrict__ grad_img,
// //                                         scalar_t* __restrict__ grad_lut,
// //                                         int H,
// //                                         int W,
// //                                         int D,
// //                                         int pixels) {
    
// //     int p = blockIdx.x * blockDim.x + threadIdx.x;

// //     if (p >= pixels) return;

// //     int hw = H * W;

// //     float r = (float)img[p];
// //     float g = (float)img[hw + p];
// //     float b = (float)img[2 * hw + p];

// //     int x0, x1, y0, y1, z0, z1;
// //     float wx0, wx1, wy0, wy1, wz0, wz1;

// //     compute_coord(b, D, x0, x1, wx0, wx1);
// //     compute_coord(g, D, y0, y1, wy0, wy1);
// //     compute_coord(r, D, z0, z1, wz0, wz1);

// //     float w000 = wz0 * wy0 * wx0;
// //     float w001 = wz0 * wy0 * wx1;
// //     float w010 = wz0 * wy1 * wx0;
// //     float w011 = wz0 * wy1 * wx1;

// //     float w100 = wz1 * wy0 * wx0;
// //     float w101 = wz1 * wy0 * wx1;
// //     float w110 = wz1 * wy1 * wx0;
// //     float w111 = wz1 * wy1 * wx1;

// //     int idx000 = lut_idx(0, z0, y0, x0, D);
// //     int idx001 = lut_idx(0, z0, y0, x1, D);
// //     int idx010 = lut_idx(0, z0, y1, x0, D);
// //     int idx011 = lut_idx(0, z0, y1, x1, D);

// //     int idx100 = lut_idx(0, z1, y0, x0, D);
// //     int idx101 = lut_idx(0, z1, y0, x1, D);
// //     int idx110 = lut_idx(0, z1, y1, x0, D);
// //     int idx111 = lut_idx(0, z1, y1, x1, D);

// //     int stride_c = D * D * D;

// //     float grad_r = 0.f;
// //     float grad_g = 0.f;
// //     float grad_b = 0.f;

// //     #pragma unroll
// //     for (int c = 0; c < 3; c++) {

// //         float go = (float)grad_out[c * hw + p];

// //         scalar_t* grad_lut_c = grad_lut + c * stride_c;

// //         atomicAdd(&grad_lut_c[idx000], go * w000);
// //         atomicAdd(&grad_lut_c[idx001], go * w001);
// //         atomicAdd(&grad_lut_c[idx010], go * w010);
// //         atomicAdd(&grad_lut_c[idx011], go * w011);

// //         atomicAdd(&grad_lut_c[idx100], go * w100);
// //         atomicAdd(&grad_lut_c[idx101], go * w101);
// //         atomicAdd(&grad_lut_c[idx110], go * w110);
// //         atomicAdd(&grad_lut_c[idx111], go * w111);

// //         const scalar_t* lut_c = lut + c * stride_c;

// //         float v000 = lut_c[idx000];
// //         float v001 = lut_c[idx001];
// //         float v010 = lut_c[idx010];
// //         float v011 = lut_c[idx011];

// //         float v100 = lut_c[idx100];
// //         float v101 = lut_c[idx101];
// //         float v110 = lut_c[idx110];
// //         float v111 = lut_c[idx111];

// //         float dx =
// //             (v001 - v000) * wz0 * wy0 +
// //             (v011 - v010) * wz0 * wy1 +
// //             (v101 - v100) * wz1 * wy0 +
// //             (v111 - v110) * wz1 * wy1;

// //         float dy =
// //             (v010 - v000) * wz0 * wx0 +
// //             (v011 - v001) * wz0 * wx1 +
// //             (v110 - v100) * wz1 * wx0 +
// //             (v111 - v101) * wz1 * wx1;

// //         float dz =
// //             (v100 - v000) * wy0 * wx0 +
// //             (v101 - v001) * wy0 * wx1 +
// //             (v110 - v010) * wy1 * wx0 +
// //             (v111 - v011) * wy1 * wx1;

// //         grad_b += go * dx;
// //         grad_g += go * dy;
// //         grad_r += go * dz;
// //     }

// //     float scale = D - 1;

// //     grad_img[p]          = (scalar_t)(grad_r * scale);
// //     grad_img[hw + p]     = (scalar_t)(grad_g * scale);
// //     grad_img[2 * hw + p] = (scalar_t)(grad_b * scale);
// // }

// template <typename scalar_t>
// __global__ void rgb_lut_backward_kernel(const scalar_t* __restrict__ grad_output,
//                                         const scalar_t* __restrict__ img,
//                                         const scalar_t* __restrict__ lut,
//                                         scalar_t* __restrict__ grad_img,
//                                         scalar_t* __restrict__ grad_lut,
//                                         int H,
//                                         int W,
//                                         int D,
//                                         int pixels) {
//     const int p = blockIdx.x * blockDim.x + threadIdx.x;
//     const int hw = H * W;

//     extern __shared__ float s_lut[];

//     const int stride = D * D * D;
//     const int lut_size = 3 * stride;

//     // 共享内存要用来累加，所以要先初始化为0，存储LUT梯度的累加值
//     for (int i = threadIdx.x; i < lut_size; i += blockDim.x) {
//         s_lut[i] = 0.f;
//     }

//     __syncthreads();

//     if (p >= pixels) {
//         return;
//     }

//     const float r = (float)img[p];
//     const float g = (float)img[hw + p];
//     const float b = (float)img[2 * hw + p];

//     const float scale = (float)(D - 1);

//     float grad_x_clip;
//     float grad_y_clip;
//     float grad_z_clip;

//     auto clip_with_grad = [&](float v, float* grad) {
//         if (isnan(v)) {
//             *grad = 0.f;
//             return 0.f;
//         }
//         if (v <= 0.f) {
//             *grad = 0.f;
//             return 0.f;
//         }
//         if (v >= scale) {
//             *grad = 0.f;
//             return scale;
//         }
//         *grad = 1.f;
//         return v;
//     };

//     const float x = clip_with_grad(b * scale, &grad_x_clip);
//     const float y = clip_with_grad(g * scale, &grad_y_clip);
//     const float z = clip_with_grad(r * scale, &grad_z_clip);

//     const int x0 = (int)x;
//     const int y0 = (int)y;
//     const int z0 = (int)z;

//     const int x1 = min(x0 + 1, D - 1);
//     const int y1 = min(y0 + 1, D - 1);
//     const int z1 = min(z0 + 1, D - 1);

//     const float wx1 = x - x0;
//     const float wy1 = y - y0;
//     const float wz1 = z - z0;

//     const float wx0 = 1.f - wx1;
//     const float wy0 = 1.f - wy1;
//     const float wz0 = 1.f - wz1;

//     const float w000 = wz0 * wy0 * wx0;
//     const float w001 = wz0 * wy0 * wx1;
//     const float w010 = wz0 * wy1 * wx0;
//     const float w011 = wz0 * wy1 * wx1;

//     const float w100 = wz1 * wy0 * wx0;
//     const float w101 = wz1 * wy0 * wx1;
//     const float w110 = wz1 * wy1 * wx0;
//     const float w111 = wz1 * wy1 * wx1;

//     const int idx000 = z0 * D * D + y0 * D + x0;
//     const int idx001 = z0 * D * D + y0 * D + x1;
//     const int idx010 = z0 * D * D + y1 * D + x0;
//     const int idx011 = z0 * D * D + y1 * D + x1;

//     const int idx100 = z1 * D * D + y0 * D + x0;
//     const int idx101 = z1 * D * D + y0 * D + x1;
//     const int idx110 = z1 * D * D + y1 * D + x0;
//     const int idx111 = z1 * D * D + y1 * D + x1;

//     float grad_r = 0.f;
//     float grad_g = 0.f;
//     float grad_b = 0.f;

//     #pragma unroll
//     for (int c = 0; c < 3; c++) {

//         const float go = (float)grad_output[c * hw + p];

//         const scalar_t* lut_c = lut + c * stride;
//         float* s_lut_c = s_lut + c * stride;

//         atomicAdd(&s_lut_c[idx000], go * w000);
//         atomicAdd(&s_lut_c[idx001], go * w001);
//         atomicAdd(&s_lut_c[idx010], go * w010);
//         atomicAdd(&s_lut_c[idx011], go * w011);

//         atomicAdd(&s_lut_c[idx100], go * w100);
//         atomicAdd(&s_lut_c[idx101], go * w101);
//         atomicAdd(&s_lut_c[idx110], go * w110);
//         atomicAdd(&s_lut_c[idx111], go * w111);

//         const float v000 = (float)lut_c[idx000];
//         const float v001 = (float)lut_c[idx001];
//         const float v010 = (float)lut_c[idx010];
//         const float v011 = (float)lut_c[idx011];

//         const float v100 = (float)lut_c[idx100];
//         const float v101 = (float)lut_c[idx101];
//         const float v110 = (float)lut_c[idx110];
//         const float v111 = (float)lut_c[idx111];

//         const float dx =
//             (v001 - v000) * wz0 * wy0 +
//             (v011 - v010) * wz0 * wy1 +
//             (v101 - v100) * wz1 * wy0 +
//             (v111 - v110) * wz1 * wy1;

//         const float dy =
//             (v010 - v000) * wz0 * wx0 +
//             (v011 - v001) * wz0 * wx1 +
//             (v110 - v100) * wz1 * wx0 +
//             (v111 - v101) * wz1 * wx1;

//         const float dz =
//             (v100 - v000) * wy0 * wx0 +
//             (v101 - v001) * wy0 * wx1 +
//             (v110 - v010) * wy1 * wx0 +
//             (v111 - v011) * wy1 * wx1;

//         grad_b += go * dx;
//         grad_g += go * dy;
//         grad_r += go * dz;
//     }

//     grad_img[p] = (scalar_t)(grad_r * grad_z_clip * scale);

//     grad_img[hw + p] = (scalar_t)(grad_g * grad_y_clip * scale);

//     grad_img[2 * hw + p] = (scalar_t)(grad_b * grad_x_clip * scale);

//     __syncthreads();

//     for (int i = threadIdx.x; i < lut_size; i += blockDim.x) {
//         atomicAdd(&grad_lut[i], (scalar_t)s_lut[i]);
//     }
// }

// } // namespace

// torch::Tensor rgb_lut_forward_cuda(torch::Tensor img, torch::Tensor lut) {
//   const c10::cuda::CUDAGuard device_guard(img.device());

//   const int64_t H = img.size(1);
//   const int64_t W = img.size(2);
//   const int64_t D = lut.size(1);
//   const int64_t pixels = H * W;
//   auto output = torch::empty_like(img);

//   const dim3 block(kThreads);
//   const dim3 grid((pixels + kThreads - 1) / kThreads);
//   auto stream = at::cuda::getCurrentCUDAStream();

//   AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "rgb_lut_forward_cuda", [&] {
//     rgb_lut_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
//         img.data_ptr<scalar_t>(),
//         lut.data_ptr<scalar_t>(),
//         output.data_ptr<scalar_t>(),
//         H,
//         W,
//         D,
//         pixels);
//   });

//   C10_CUDA_KERNEL_LAUNCH_CHECK();
//   return output;
// }

// std::vector<torch::Tensor> rgb_lut_backward_cuda(torch::Tensor grad_output, torch::Tensor img, torch::Tensor lut) {
//   const c10::cuda::CUDAGuard device_guard(img.device());

//   const int64_t H = img.size(1);
//   const int64_t W = img.size(2);
//   const int64_t D = lut.size(1);
//   const int64_t pixels = H * W;
//   auto grad_img = torch::empty_like(img);
//   auto grad_lut = torch::zeros_like(lut);

//   const dim3 block(kThreads);
//   const dim3 grid((pixels + kThreads - 1) / kThreads);
//   auto stream = at::cuda::getCurrentCUDAStream();

//   int lut_size = 3 * D * D * D;
//   size_t shared_mem = lut_size * sizeof(float);

//   AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "rgb_lut_backward_cuda", [&] {
//     rgb_lut_backward_kernel<scalar_t><<<grid, block, shared_mem, stream>>>(
//         grad_output.data_ptr<scalar_t>(),
//         img.data_ptr<scalar_t>(),
//         lut.data_ptr<scalar_t>(),
//         grad_img.data_ptr<scalar_t>(),
//         grad_lut.data_ptr<scalar_t>(),
//         H,
//         W,
//         D,
//         pixels);
//   });

//   C10_CUDA_KERNEL_LAUNCH_CHECK();
//   return {grad_img, grad_lut};
// }


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
