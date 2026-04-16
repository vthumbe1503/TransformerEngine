/*
 * Memory-bandwidth benchmark: C[i] = A[i] + B[i]
 *
 * Uses 128-bit (float4) vectorized loads/stores with grid-stride loops
 * and large-page allocations via the CUDA driver API.
 *
 * Optimized variant uses streaming (non-caching) loads (__ldcs) and
 * streaming stores (__stcs) to bypass L2 and eliminate read-for-ownership
 * traffic on the output array.
 *
 * Build:
 *   nvcc -O3 -arch=sm_100 -o bench_vecadd bench_vecadd_bandwidth.cu -lcuda
 *
 * Run:
 *   ./bench_vecadd --dtype bf16
 *   ./bench_vecadd --dtype bf16 --streaming
 *   ./bench_vecadd --dtype bf16 --compare-all
 *   ./bench_vecadd --dtype bf16 --size 536870912
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA RT error at %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_DRV(call)                                                        \
  do {                                                                         \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *s = nullptr;                                                 \
      cuGetErrorString(err, &s);                                               \
      fprintf(stderr, "CUDA DRV error at %s:%d: %s\n", __FILE__, __LINE__,    \
              s ? s : "unknown");                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ─── Large-page allocation via CUDA driver API ────────────────────────────────

struct LargePageAlloc {
  CUdeviceptr ptr;
  CUmemGenericAllocationHandle handle;
  size_t alloc_size;
};

static size_t get_granularity(int dev) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  size_t gran = 0;
  CHECK_DRV(cuMemGetAllocationGranularity(
      &gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  return gran;
}

static LargePageAlloc large_page_alloc(size_t size, int dev) {
  size_t gran = get_granularity(dev);
  size_t alloc_size = ((size + gran - 1) / gran) * gran;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;

  CUmemGenericAllocationHandle handle;
  CHECK_DRV(cuMemCreate(&handle, alloc_size, &prop, 0));

  CUdeviceptr ptr = 0;
  CHECK_DRV(cuMemAddressReserve(&ptr, alloc_size, gran, 0, 0));
  CHECK_DRV(cuMemMap(ptr, alloc_size, 0, handle, 0));

  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = dev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_DRV(cuMemSetAccess(ptr, alloc_size, &access, 1));

  return {ptr, handle, alloc_size};
}

static void large_page_free(LargePageAlloc &a) {
  CHECK_DRV(cuMemUnmap(a.ptr, a.alloc_size));
  CHECK_DRV(cuMemRelease(a.handle));
  CHECK_DRV(cuMemAddressFree(a.ptr, a.alloc_size));
}

// ─── Streaming load/store helpers ─────────────────────────────────────────────
//
// __ldcs: load with cache-streaming hint  (L2 bypass / evict-first)
// __stcs: store with cache-streaming hint (skips read-for-ownership)

__device__ __forceinline__ float4 ldcs(const float4 *p) {
  return __ldcs(p);
}

__device__ __forceinline__ void stcs(float4 *p, float4 v) {
  __stcs(p, v);
}

// ─── DEFAULT kernels: cached loads, cached stores ─────────────────────────────

__global__ void vecadd_f32_kernel(const float4 *__restrict__ A,
                                  const float4 *__restrict__ B,
                                  float4 *__restrict__ C,
                                  size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = A[idx];
    float4 b = B[idx];
    C[idx] = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  }
}

__global__ void vecadd_f16_kernel(const float4 *__restrict__ A,
                                  const float4 *__restrict__ B,
                                  float4 *__restrict__ C,
                                  size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = A[idx];
    float4 b = B[idx];
    __half2 *ha = reinterpret_cast<__half2 *>(&a);
    __half2 *hb = reinterpret_cast<__half2 *>(&b);
    float4 c;
    __half2 *hc = reinterpret_cast<__half2 *>(&c);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    C[idx] = c;
  }
}

__global__ void vecadd_bf16_kernel(const float4 *__restrict__ A,
                                   const float4 *__restrict__ B,
                                   float4 *__restrict__ C,
                                   size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = A[idx];
    float4 b = B[idx];
    __nv_bfloat162 *ha = reinterpret_cast<__nv_bfloat162 *>(&a);
    __nv_bfloat162 *hb = reinterpret_cast<__nv_bfloat162 *>(&b);
    float4 c;
    __nv_bfloat162 *hc = reinterpret_cast<__nv_bfloat162 *>(&c);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    C[idx] = c;
  }
}

// ─── STREAMING kernels: __ldcs loads + __stcs stores (L2 bypass, no RFO) ─────

__global__ void vecadd_f32_streaming_kernel(const float4 *__restrict__ A,
                                            const float4 *__restrict__ B,
                                            float4 *__restrict__ C,
                                            size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = ldcs(A + idx);
    float4 b = ldcs(B + idx);
    stcs(C + idx, {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w});
  }
}

__global__ void vecadd_f16_streaming_kernel(const float4 *__restrict__ A,
                                            const float4 *__restrict__ B,
                                            float4 *__restrict__ C,
                                            size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = ldcs(A + idx);
    float4 b = ldcs(B + idx);
    __half2 *ha = reinterpret_cast<__half2 *>(&a);
    __half2 *hb = reinterpret_cast<__half2 *>(&b);
    float4 c;
    __half2 *hc = reinterpret_cast<__half2 *>(&c);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    stcs(C + idx, c);
  }
}

__global__ void vecadd_bf16_streaming_kernel(const float4 *__restrict__ A,
                                             const float4 *__restrict__ B,
                                             float4 *__restrict__ C,
                                             size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = ldcs(A + idx);
    float4 b = ldcs(B + idx);
    __nv_bfloat162 *ha = reinterpret_cast<__nv_bfloat162 *>(&a);
    __nv_bfloat162 *hb = reinterpret_cast<__nv_bfloat162 *>(&b);
    float4 c;
    __nv_bfloat162 *hc = reinterpret_cast<__nv_bfloat162 *>(&c);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    stcs(C + idx, c);
  }
}

// ─── HYBRID kernels: cached loads + streaming stores (best for large arrays) ─

__global__ void vecadd_f32_hybrid_kernel(const float4 *__restrict__ A,
                                         const float4 *__restrict__ B,
                                         float4 *__restrict__ C,
                                         size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = A[idx];
    float4 b = B[idx];
    stcs(C + idx, {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w});
  }
}

__global__ void vecadd_f16_hybrid_kernel(const float4 *__restrict__ A,
                                         const float4 *__restrict__ B,
                                         float4 *__restrict__ C,
                                         size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = A[idx];
    float4 b = B[idx];
    __half2 *ha = reinterpret_cast<__half2 *>(&a);
    __half2 *hb = reinterpret_cast<__half2 *>(&b);
    float4 c;
    __half2 *hc = reinterpret_cast<__half2 *>(&c);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    stcs(C + idx, c);
  }
}

__global__ void vecadd_bf16_hybrid_kernel(const float4 *__restrict__ A,
                                          const float4 *__restrict__ B,
                                          float4 *__restrict__ C,
                                          size_t n_vec) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < n_vec; idx += stride) {
    float4 a = A[idx];
    float4 b = B[idx];
    __nv_bfloat162 *ha = reinterpret_cast<__nv_bfloat162 *>(&a);
    __nv_bfloat162 *hb = reinterpret_cast<__nv_bfloat162 *>(&b);
    float4 c;
    __nv_bfloat162 *hc = reinterpret_cast<__nv_bfloat162 *>(&c);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    stcs(C + idx, c);
  }
}

// ─── Launch helper ────────────────────────────────────────────────────────────

enum Dtype { FP32, FP16, BF16 };
enum StoreMode { CACHED, STREAMING, HYBRID };

static constexpr int THREADS_PER_BLOCK = 1024;

void launch_vecadd(const void *A, const void *B, void *C,
                   size_t n_bytes, Dtype dt, cudaStream_t stream,
                   int num_sms, int blocks_per_sm, StoreMode mode) {
  size_t n_vec = n_bytes / 16;
  int blocks = num_sms * blocks_per_sm;

  const float4 *vA = reinterpret_cast<const float4 *>(A);
  const float4 *vB = reinterpret_cast<const float4 *>(B);
  float4 *vC       = reinterpret_cast<float4 *>(C);

  switch (mode) {
    case STREAMING:
      switch (dt) {
        case FP32: vecadd_f32_streaming_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
        case FP16: vecadd_f16_streaming_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
        case BF16: vecadd_bf16_streaming_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
      }
      break;
    case HYBRID:
      switch (dt) {
        case FP32: vecadd_f32_hybrid_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
        case FP16: vecadd_f16_hybrid_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
        case BF16: vecadd_bf16_hybrid_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
      }
      break;
    default:
      switch (dt) {
        case FP32: vecadd_f32_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
        case FP16: vecadd_f16_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
        case BF16: vecadd_bf16_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vA, vB, vC, n_vec); break;
      }
      break;
  }
}

// ─── Benchmark driver ─────────────────────────────────────────────────────────

struct BenchConfig {
  int warmup;
  int iters;
  int num_sms;
  int dev;
  int blocks_per_sm;
  StoreMode mode;
};

static const char *mode_name(StoreMode m) {
  switch (m) {
    case CACHED:    return "default";
    case STREAMING: return "streaming";
    case HYBRID:    return "hybrid";
  }
  return "unknown";
}

static const char *mode_desc(StoreMode m) {
  switch (m) {
    case CACHED:    return "DEFAULT (cached ld + cached st)";
    case STREAMING: return "STREAMING (__ldcs + __stcs, full L2 bypass)";
    case HYBRID:    return "HYBRID (cached ld + __stcs, no RFO on stores)";
  }
  return "unknown";
}

double bench_one(size_t n_elems, size_t elem_size, Dtype dt,
                 const BenchConfig &cfg, bool use_large_pages) {
  size_t total_bytes = n_elems * elem_size;
  total_bytes = (total_bytes / 16) * 16;
  size_t traffic = 3ULL * total_bytes;

  void *dA_ptr, *dB_ptr, *dC_ptr;
  LargePageAlloc lpA, lpB, lpC;

  if (use_large_pages) {
    lpA = large_page_alloc(total_bytes, cfg.dev);
    lpB = large_page_alloc(total_bytes, cfg.dev);
    lpC = large_page_alloc(total_bytes, cfg.dev);
    dA_ptr = reinterpret_cast<void *>(lpA.ptr);
    dB_ptr = reinterpret_cast<void *>(lpB.ptr);
    dC_ptr = reinterpret_cast<void *>(lpC.ptr);
  } else {
    CHECK_CUDA(cudaMalloc(&dA_ptr, total_bytes));
    CHECK_CUDA(cudaMalloc(&dB_ptr, total_bytes));
    CHECK_CUDA(cudaMalloc(&dC_ptr, total_bytes));
  }
  CHECK_CUDA(cudaMemset(dA_ptr, 1, total_bytes));
  CHECK_CUDA(cudaMemset(dB_ptr, 1, total_bytes));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < cfg.warmup; ++i)
    launch_vecadd(dA_ptr, dB_ptr, dC_ptr, total_bytes, dt, stream,
                  cfg.num_sms, cfg.blocks_per_sm, cfg.mode);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < cfg.iters; ++i)
    launch_vecadd(dA_ptr, dB_ptr, dC_ptr, total_bytes, dt, stream,
                  cfg.num_sms, cfg.blocks_per_sm, cfg.mode);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  double avg_s  = (ms / 1000.0) / cfg.iters;
  double bw_tbs = (double)traffic / avg_s / 1e12;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));

  if (use_large_pages) {
    large_page_free(lpA);
    large_page_free(lpB);
    large_page_free(lpC);
  } else {
    CHECK_CUDA(cudaFree(dA_ptr));
    CHECK_CUDA(cudaFree(dB_ptr));
    CHECK_CUDA(cudaFree(dC_ptr));
  }

  return bw_tbs;
}

double bench_memcpy(size_t total_bytes, int warmup, int iters, int dev,
                    bool use_large_pages) {
  total_bytes = (total_bytes / 16) * 16;

  void *dSrc, *dDst;
  LargePageAlloc lpSrc, lpDst;

  if (use_large_pages) {
    lpSrc = large_page_alloc(total_bytes, dev);
    lpDst = large_page_alloc(total_bytes, dev);
    dSrc = reinterpret_cast<void *>(lpSrc.ptr);
    dDst = reinterpret_cast<void *>(lpDst.ptr);
  } else {
    CHECK_CUDA(cudaMalloc(&dSrc, total_bytes));
    CHECK_CUDA(cudaMalloc(&dDst, total_bytes));
  }
  CHECK_CUDA(cudaMemset(dSrc, 1, total_bytes));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i)
    CHECK_CUDA(cudaMemcpyAsync(dDst, dSrc, total_bytes,
                               cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i)
    CHECK_CUDA(cudaMemcpyAsync(dDst, dSrc, total_bytes,
                               cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  double avg_s = (ms / 1000.0) / iters;
  double bw_tbs = (2.0 * total_bytes) / avg_s / 1e12;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));

  if (use_large_pages) {
    large_page_free(lpSrc);
    large_page_free(lpDst);
  } else {
    CHECK_CUDA(cudaFree(dSrc));
    CHECK_CUDA(cudaFree(dDst));
  }
  return bw_tbs;
}

static double compute_peak_bw(int dev) {
  int clock_khz = 0, bus_bits = 0;
  cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
  cuDeviceGetAttribute(&bus_bits, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
  if (clock_khz == 0 || bus_bits == 0) return 0.0;
  double clock_hz = clock_khz * 1e3;
  double bus_bytes = bus_bits / 8.0;
  return 2.0 * clock_hz * bus_bytes / 1e12;
}

void print_device_info(int num_sms, int blocks_per_sm, bool use_large_pages,
                       StoreMode mode, double peak_bw) {
  int dev;
  CHECK_CUDA(cudaGetDevice(&dev));

  char name[256] = {};
  cuDeviceGetName(name, sizeof(name), dev);

  int sm_count = 0;
  cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

  size_t total_mem = 0;
  cuDeviceTotalMem(&total_mem, dev);

  int bus_bits = 0;
  cuDeviceGetAttribute(&bus_bits, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);

  size_t gran = get_granularity(dev);
  int blocks = num_sms * blocks_per_sm;

  printf("════════════════════════════════════════════════════════════════\n");
  printf("  GPU            : %s\n", name);
  printf("  SMs            : %d\n", sm_count);
  printf("  Global memory  : %.1f GiB\n",
         total_mem / (1024.0 * 1024 * 1024));
  printf("  Bus width      : %d bit\n", bus_bits);
  printf("  Theo. peak BW  : ~%.1f TB/s\n", peak_bw);
  printf("  Vec load width : 128 bit (float4)\n");
  printf("  Load/store     : %s\n", mode_desc(mode));
  printf("  Page size      : %s (%zu bytes)\n",
         use_large_pages ? "LARGE (driver API)" : "default (4 KB)",
         use_large_pages ? gran : 4096UL);
  printf("  Grid config    : %d blocks × %d threads (%d blks/SM, grid-stride)\n",
         blocks, THREADS_PER_BLOCK, blocks_per_sm);
  printf("════════════════════════════════════════════════════════════════\n\n");
}

static const char *fmt_bytes(size_t b) {
  static char buf[4][32];
  static int slot = 0;
  char *p = buf[slot++ & 3];
  if (b >= (1ULL << 30))
    snprintf(p, 32, "%.1f GiB", b / (double)(1ULL << 30));
  else if (b >= (1ULL << 20))
    snprintf(p, 32, "%.1f MiB", b / (double)(1ULL << 20));
  else
    snprintf(p, 32, "%.1f KiB", b / (double)(1ULL << 10));
  return p;
}

int main(int argc, char **argv) {
  BenchConfig cfg = {100, 500, 0, 0, 4, CACHED};
  size_t single_size = 0;
  const char *dtype_str = "fp32";
  bool use_large_pages = true;
  bool compare_pages = false;
  bool compare_all = false;
  bool sweep_grid = false;
  bool run_memcpy = false;
  double user_peak_bw = 0.0;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
      cfg.warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      cfg.iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--size") && i + 1 < argc)
      single_size = (size_t)atoll(argv[++i]);
    else if (!strcmp(argv[i], "--dtype") && i + 1 < argc)
      dtype_str = argv[++i];
    else if (!strcmp(argv[i], "--no-large-pages"))
      use_large_pages = false;
    else if (!strcmp(argv[i], "--streaming"))
      cfg.mode = STREAMING;
    else if (!strcmp(argv[i], "--hybrid"))
      cfg.mode = HYBRID;
    else if (!strcmp(argv[i], "--blocks-per-sm") && i + 1 < argc)
      cfg.blocks_per_sm = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--compare"))
      compare_pages = true;
    else if (!strcmp(argv[i], "--compare-all"))
      compare_all = true;
    else if (!strcmp(argv[i], "--sweep-grid"))
      sweep_grid = true;
    else if (!strcmp(argv[i], "--memcpy"))
      run_memcpy = true;
    else if (!strcmp(argv[i], "--peak-bw") && i + 1 < argc)
      user_peak_bw = atof(argv[++i]);
    else if (!strcmp(argv[i], "--help")) {
      printf("Usage: %s [--size N] [--dtype fp32|fp16|bf16] "
             "[--warmup W] [--iters I]\n"
             "       [--no-large-pages] [--streaming] [--hybrid] "
             "[--blocks-per-sm N]\n"
             "       [--compare] [--compare-all] [--sweep-grid] "
             "[--peak-bw TB/s]\n", argv[0]);
      printf("\n");
      printf("  --streaming      __ldcs + __stcs (full L2 bypass)\n");
      printf("  --hybrid         cached ld + __stcs (no RFO on stores only)\n");
      printf("  --blocks-per-sm  Blocks per SM (default: 4)\n");
      printf("  --compare        Compare 4KB vs large pages\n");
      printf("  --compare-all    Compare all 6 combos: "
             "{4KB,large} × {default,streaming,hybrid}\n");
      printf("  --sweep-grid     Sweep blocks-per-SM {1,2,4,8,16,32} "
             "at a fixed size\n");
      printf("  --memcpy         Run cudaMemcpy D2D baseline (NVIDIA's "
             "optimized copy)\n");
      printf("  --peak-bw N      Override theoretical peak BW in TB/s\n");
      return 0;
    }
  }

  CHECK_DRV(cuInit(0));

  int dev;
  CHECK_CUDA(cudaGetDevice(&dev));
  int sm_count = 0;
  cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  cfg.num_sms = sm_count;
  cfg.dev = dev;

  double peak_bw = (user_peak_bw > 0) ? user_peak_bw : compute_peak_bw(dev);

  Dtype dt;
  size_t elem_size;
  if (!strcmp(dtype_str, "fp32")) {
    dt = FP32; elem_size = 4;
  } else if (!strcmp(dtype_str, "fp16")) {
    dt = FP16; elem_size = 2;
  } else if (!strcmp(dtype_str, "bf16")) {
    dt = BF16; elem_size = 2;
  } else {
    fprintf(stderr, "Unknown dtype: %s (use fp32, fp16, bf16)\n", dtype_str);
    return 1;
  }

  size_t sweep[] = {
      256ULL * 1024,
      1ULL * 1024 * 1024,
      4ULL * 1024 * 1024,
      16ULL * 1024 * 1024,
      64ULL * 1024 * 1024,
      128ULL * 1024 * 1024,
      256ULL * 1024 * 1024,
      512ULL * 1024 * 1024,
      1024ULL * 1024 * 1024,
      2048ULL * 1024 * 1024,
      4096ULL * 1024 * 1024,
  };
  int n_sweep = sizeof(sweep) / sizeof(sweep[0]);

  auto run_suite = [&](bool large_pages, StoreMode mode, int bpsm) {
    cfg.mode = mode;
    cfg.blocks_per_sm = bpsm;
    print_device_info(cfg.num_sms, bpsm, large_pages, mode, peak_bw);

    printf("dtype: %s  |  pages: %s  |  ld/st: %s  |  blks/SM: %d  |  "
           "warmup: %d  |  iters: %d\n",
           dtype_str,
           large_pages ? "LARGE" : "4KB",
           mode_name(mode), bpsm,
           cfg.warmup, cfg.iters);
    printf("──────────────────────────────────────────────────────────────────\n");
    printf("  %14s  %14s  %14s  %12s  %8s\n",
           "Elements", "Per-array", "Total traffic", "Bandwidth", "% Peak");
    printf("──────────────────────────────────────────────────────────────────\n");

    auto bench_and_print = [&](size_t n) {
      size_t per_array = n * elem_size;
      size_t traffic   = 3 * per_array;
      double bw = bench_one(n, elem_size, dt, cfg, large_pages);
      double pct = 100.0 * bw / peak_bw;
      printf("  %14zu  %14s  %14s  %9.2f TB/s  %6.1f%%\n",
             n, fmt_bytes(per_array), fmt_bytes(traffic), bw, pct);
      fflush(stdout);
    };

    if (single_size > 0) {
      bench_and_print(single_size);
    } else {
      for (int i = 0; i < n_sweep; ++i) {
        size_t n = sweep[i];
        if (n * elem_size * 3 > 48ULL * 1024 * 1024 * 1024) break;
        bench_and_print(n);
      }
    }
    printf("\n");
  };

  if (run_memcpy) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  cudaMemcpy D2D BASELINE (NVIDIA driver-optimized copy)    ║\n");
    printf("║  Traffic = 2 × size (1 read + 1 write, no compute)         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  %14s  %14s  %12s  %8s\n",
           "Size", "Traffic (R+W)", "Bandwidth", "% Peak");
    printf("  ──────────────  ──────────────  ────────────  ────────\n");

    for (int i = 0; i < n_sweep; ++i) {
      size_t bytes = sweep[i] * elem_size;
      if (bytes * 2 > 48ULL * 1024 * 1024 * 1024) break;
      double bw = bench_memcpy(bytes, cfg.warmup, cfg.iters, cfg.dev,
                               use_large_pages);
      double pct = 100.0 * bw / peak_bw;
      printf("  %14s  %14s  %9.2f TB/s  %6.1f%%\n",
             fmt_bytes(bytes), fmt_bytes(2 * bytes), bw, pct);
      fflush(stdout);
    }
    printf("\n");
    return 0;
  }

  if (sweep_grid) {
    size_t sweep_size = (single_size > 0) ? single_size
                                          : 512ULL * 1024 * 1024;
    int bpsm_vals[] = {1, 2, 4, 8, 16, 32};
    StoreMode modes[] = {CACHED, HYBRID, STREAMING};

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  GRID SWEEP: %zu elems (%s per array), large_pages=%s      \n",
           sweep_size, fmt_bytes(sweep_size * elem_size),
           use_large_pages ? "yes" : "no");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  %16s  %8s  %12s  %12s  %8s\n",
           "Mode", "Blks/SM", "Tot blocks", "Bandwidth", "% Peak");
    printf("  ────────────────  ────────  ────────────  ────────────  ────────\n");

    for (auto m : modes) {
      for (auto b : bpsm_vals) {
        cfg.mode = m;
        cfg.blocks_per_sm = b;
        double bw = bench_one(sweep_size, elem_size, dt, cfg, use_large_pages);
        double pct = 100.0 * bw / peak_bw;
        printf("  %16s  %8d  %12d  %9.2f TB/s  %6.1f%%\n",
               mode_name(m), b, cfg.num_sms * b, bw, pct);
        fflush(stdout);
      }
      printf("  ────────────────  ────────  ────────────  ────────────  ────────\n");
    }
    printf("\n");
  } else if (compare_all) {
    StoreMode modes[] = {CACHED, STREAMING, HYBRID};
    for (auto m : modes) {
      run_suite(false, m, cfg.blocks_per_sm);
      run_suite(true,  m, cfg.blocks_per_sm);
    }
  } else if (compare_pages) {
    run_suite(false, cfg.mode, cfg.blocks_per_sm);
    run_suite(true,  cfg.mode, cfg.blocks_per_sm);
  } else {
    run_suite(use_large_pages, cfg.mode, cfg.blocks_per_sm);
  }

  return 0;
}
