// sam2/csrc/connected_components.cu
// ===================================
// CUDA kernel for 2-D connected-component labelling.
// Used by fill_holes_in_mask_scores() post-processing in sam2/utils/misc.py.
//
// Adapted from https://github.com/zsef123/Connected_components_PyTorch
// (MIT license; see LICENSE_cctorch in the repository root).

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <vector>

#define BLOCK_ROWS 16
#define BLOCK_COLS 16

namespace cc2d {

template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
  return (bitmap >> pos) & 1;
}

__device__ int32_t find(const int32_t* s_buf, int32_t n) {
  while (s_buf[n] != n) n = s_buf[n];
  return n;
}

__device__ int32_t find_n_compress(int32_t* s_buf, int32_t n) {
  const int32_t id = n;
  while (s_buf[n] != n) { n = s_buf[n]; s_buf[id] = n; }
  return n;
}

__device__ void union_(int32_t* s_buf, int32_t a, int32_t b) {
  bool done;
  do {
    a = find(s_buf, a); b = find(s_buf, b);
    if      (a < b) { int32_t old = atomicMin(s_buf + b, a); done = (old == b); b = old; }
    else if (b < a) { int32_t old = atomicMin(s_buf + a, b); done = (old == a); a = old; }
    else               done = true;
  } while (!done);
}

__global__ void init_labeling(int32_t* label, const uint32_t W, const uint32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (row < H && col < W) label[row * W + col] = row * W + col;
}

__global__ void merge(uint8_t* img, int32_t* label, const uint32_t W, const uint32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
  if (row >= H || col >= W) return;
  uint32_t P = 0;
  if (img[idx])              P |= 0x777;
  if (row+1<H&&img[idx+W])  P |= 0x777 << 4;
  if (col+1<W&&img[idx+1])  P |= 0x777 << 1;
  if (col==0)   P &= 0xEEEE;
  if (col+1>=W) P &= 0x3333; else if (col+2>=W) P &= 0x7777;
  if (row==0)   P &= 0xFFF0;
  if (row+1>=H) P &= 0xFF;
  if (P > 0) {
    if (hasBit(P,0)&&img[idx-W-1])              union_(label,idx,idx-2*W-2);
    if ((hasBit(P,1)&&img[idx-W])||(hasBit(P,2)&&img[idx-W+1])) union_(label,idx,idx-2*W);
    if (hasBit(P,3)&&img[idx+2-W])              union_(label,idx,idx-2*W+2);
    if ((hasBit(P,4)&&img[idx-1])||(hasBit(P,8)&&img[idx+W-1])) union_(label,idx,idx-2);
  }
}

__global__ void compression(int32_t* label, const int32_t W, const int32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (row < H && col < W) find_n_compress(label, row * W + col);
}

__global__ void final_labeling(const uint8_t* img, int32_t* label, const int32_t W, const int32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
  if (row >= H || col >= W) return;
  int32_t y = label[idx] + 1;
  label[idx] = img[idx] ? y : 0;
  if (col+1<W) { label[idx+1] = img[idx+1] ? y : 0;
    if (row+1<H) label[idx+W+1] = img[idx+W+1] ? y : 0; }
  if (row+1<H)   label[idx+W] = img[idx+W] ? y : 0;
}

__global__ void init_counting(const int32_t* label, int32_t* count_init, const int32_t W, const int32_t H) {
  const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= H || col >= W) return;
  int32_t y = label[row * W + col];
  if (y > 0) atomicAdd(count_init + y - 1, 1);
}

__global__ void final_counting(const int32_t* label, const int32_t* count_init, int32_t* count_final,
                                const int32_t W, const int32_t H) {
  const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= H || col >= W) return;
  int32_t y = label[row * W + col];
  count_final[row * W + col] = (y > 0) ? count_init[y - 1] : 0;
}

} // namespace cc2d

std::vector<torch::Tensor> get_connected_componnets(const torch::Tensor& inputs) {
  AT_ASSERTM(inputs.is_cuda(), "inputs must be a CUDA tensor");
  AT_ASSERTM(inputs.ndimension() == 4, "inputs must be [N,1,H,W]");
  AT_ASSERTM(inputs.scalar_type() == torch::kUInt8, "inputs must be uint8");
  const uint32_t N = inputs.size(0), C = inputs.size(1), H = inputs.size(2), W = inputs.size(3);
  AT_ASSERTM(C == 1, "inputs must be [N,1,H,W]");
  AT_ASSERTM((H % 2) == 0 && (W % 2) == 0, "H and W must be even");

  auto lo = torch::TensorOptions().dtype(torch::kInt32).device(inputs.device());
  auto labels       = torch::zeros({N,C,H,W}, lo);
  auto counts_init  = torch::zeros({N,C,H,W}, lo);
  auto counts_final = torch::zeros({N,C,H,W}, lo);

  dim3 grid(((W+1)/2+BLOCK_COLS-1)/BLOCK_COLS, ((H+1)/2+BLOCK_ROWS-1)/BLOCK_ROWS);
  dim3 block(BLOCK_COLS, BLOCK_ROWS);
  dim3 grid_cnt((W+BLOCK_COLS)/BLOCK_COLS, (H+BLOCK_ROWS)/BLOCK_ROWS);
  dim3 block_cnt(BLOCK_COLS, BLOCK_ROWS);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int n = 0; n < (int)N; n++) {
    uint32_t off = n * H * W;
    cc2d::init_labeling<<<grid,block,0,stream>>>(labels.data_ptr<int32_t>()+off, W, H);
    cc2d::merge<<<grid,block,0,stream>>>(inputs.data_ptr<uint8_t>()+off, labels.data_ptr<int32_t>()+off, W, H);
    cc2d::compression<<<grid,block,0,stream>>>(labels.data_ptr<int32_t>()+off, W, H);
    cc2d::final_labeling<<<grid,block,0,stream>>>(inputs.data_ptr<uint8_t>()+off, labels.data_ptr<int32_t>()+off, W, H);
    cc2d::init_counting<<<grid_cnt,block_cnt,0,stream>>>(labels.data_ptr<int32_t>()+off, counts_init.data_ptr<int32_t>()+off, W, H);
    cc2d::final_counting<<<grid_cnt,block_cnt,0,stream>>>(labels.data_ptr<int32_t>()+off, counts_init.data_ptr<int32_t>()+off, counts_final.data_ptr<int32_t>()+off, W, H);
  }
  return {labels, counts_final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_connected_componnets", &get_connected_componnets, "get_connected_componnets");
}
